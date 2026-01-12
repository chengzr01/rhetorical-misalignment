from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
import json
import os
from datetime import datetime
import hashlib

# Import from our modules
from config import DATASETS, AVAILABLE_MODELS
from data_loader import (
    get_dataset_config,
    get_model_info,
    get_available_models,
    check_dataset_availability,
    load_data,
    load_manipulative_case_ids,
    get_case_file_model_name,
    get_indices_for_case_ids,
    parse_indices,
    get_random_indices
)
from scheduler import (
    get_annotation_counts_per_case,
    get_smart_random_cases,
    select_model_with_fewest_annotations,
    get_coverage_statistics
)
from annotation_utils import (
    render_markdown,
    save_annotation
)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

# Ensure annotation directories exist
for dataset_key, dataset_config in DATASETS.items():
    os.makedirs(dataset_config['annotation_dir'], exist_ok=True)


@app.route('/')
def index():
    """Landing page - redirect based on consent status"""
    # Check if user has already consented
    if session.get('consent_given', False):
        return redirect(url_for('demographics'))
    else:
        return redirect(url_for('consent'))

@app.route('/consent')
def consent():
    """Standalone consent form page"""
    return render_template('consent_form.html')

@app.route('/files/<path:filename>')
def serve_file(filename):
    """Serve files from the files directory"""
    files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
    return send_from_directory(files_dir, filename)

@app.route('/consent_submit', methods=['POST'])
def consent_submit():
    """Handle consent form submission"""
    consent_value = request.form.get('consent', '')

    if consent_value == 'agree':
        session['consent_given'] = True
        session['consent_timestamp'] = datetime.now().isoformat()
        return redirect(url_for('demographics'))
    else:
        # User did not agree - show message and stay on consent page
        return render_template('consent_form.html', declined=True)

@app.route('/demographics')
def demographics():
    """Demographics and study information page"""
    # Check if user has consented
    consent_given = session.get('consent_given', False)

    # Check dataset availability
    dataset_availability = check_dataset_availability()

    # Find first available dataset
    default_dataset = 'usmle_sample'
    for dataset_key in ['usmle_sample', 'usmle', 'mimic']:
        if dataset_availability.get(dataset_key, False):
            default_dataset = dataset_key
            break

    # Get available models for default dataset
    available_models = get_available_models(default_dataset)

    # Get default model to show total cases
    available_model_keys = [m['key'] for m in available_models if m.get('available', False)]
    default_model = available_model_keys[0] if available_model_keys else 'llama_small'

    # Try to load data to get case count
    total_cases = 0
    try:
        data = load_data(default_model, default_dataset)
        total_cases = len(data)
    except:
        pass

    return render_template('demographics.html',
                          total_cases=total_cases,
                          available_models=available_models,
                          datasets=DATASETS,
                          dataset_availability=dataset_availability,
                          consent_given=consent_given)

@app.route('/api/models/<dataset_key>')
def api_get_models(dataset_key):
    """API endpoint to get available models for a dataset"""
    models = get_available_models(dataset_key)
    return jsonify({'models': models})

@app.route('/api/coverage/<dataset_key>/<model_key>')
def api_get_coverage(dataset_key, model_key):
    """API endpoint to get annotation coverage statistics for a dataset and model"""
    stats = get_coverage_statistics(dataset_key, model_key)
    return jsonify(stats)

@app.route('/start', methods=['POST'])
def start_annotation():
    """Start annotation session"""
    # Check if user has consented
    if not session.get('consent_given', False):
        return redirect(url_for('consent'))

    annotator_id = request.form.get('annotator_id', 'anonymous')
    dataset_key = request.form.get('dataset', 'mimic')
    selection_mode = request.form.get('selection_mode', 'all')

    # Check if automatic model balancing is enabled
    auto_balance_models = request.form.get('auto_balance_models', 'on') == 'on'

    if auto_balance_models:
        # Automatically select the model with fewest annotations
        model_key = select_model_with_fewest_annotations(dataset_key)
        print(f"Auto-selected model: {model_key} for balanced annotation coverage")
    else:
        # Use manually selected model
        model_key = request.form.get('model_key', 'llama_small')
        print(f"Manually selected model: {model_key}")

    # Collect demographic information
    demographics = {
        'expertise': request.form.get('expertise', ''),
        'years_of_practice': request.form.get('years_of_practice', ''),
        'age': request.form.get('age', ''),
        'sex': request.form.get('sex', ''),
        'race': request.form.get('race', ''),
        'practice_location': request.form.get('practice_location', ''),
        'submitted_at': datetime.now().isoformat()
    }

    # Load data for selected model and dataset
    data = load_data(model_key, dataset_key)
    total_cases = len(data)

    if selection_mode == 'targeted':
        # Get targeted case indices
        targeted_type = request.form.get('targeted_type', 'indices')

        if targeted_type == 'indices':
            # Parse specific indices
            indices_string = request.form.get('case_indices', '')
            case_indices = parse_indices(indices_string, total_cases)
            if not case_indices:
                # If no valid indices, default to all cases
                case_indices = list(range(total_cases))
        elif targeted_type == 'manipulative':
            # Load manipulative cases for this model and dataset
            manipulative_case_ids = load_manipulative_case_ids(model_key, dataset_key)

            if not manipulative_case_ids:
                # Fallback to all cases if no manipulative cases found
                case_indices = list(range(total_cases))
            else:
                # Get annotation counts for smart selection (model-specific)
                annotation_counts = get_annotation_counts_per_case(dataset_key, model_key)

                # Smart selection: prioritize cases with fewer annotations
                # Randomly select 10 cases, prioritizing those with 0 annotations, then 1, then 2, then 3+
                selected_case_ids = get_smart_random_cases(
                    manipulative_case_ids,
                    annotation_counts,
                    num_cases=10
                )

                # Convert selected case IDs to indices
                case_indices = get_indices_for_case_ids(data, selected_case_ids)
        else:
            # Random sampling
            random_count = int(request.form.get('random_count', 10))
            case_indices = get_random_indices(total_cases, random_count)
    else:
        # All cases in order
        case_index = int(request.form.get('case_index', 0))
        case_indices = list(range(case_index, total_cases))

    # Apply step filter if specified
    selected_steps = request.form.getlist('step_filter')
    if selected_steps:
        # Filter case_indices to only include cases with selected meta_info values
        filtered_indices = []
        for idx in case_indices:
            case = data[idx]
            case_meta_info = case.get('meta_info', '')
            if case_meta_info in selected_steps:
                filtered_indices.append(idx)
        case_indices = filtered_indices if filtered_indices else case_indices

    session['annotator_id'] = annotator_id if annotator_id else 'anonymous'
    session['demographics'] = demographics
    session['dataset_key'] = dataset_key
    session['model_key'] = model_key
    session['case_indices'] = case_indices
    session['current_position'] = 0
    session['annotated_cases'] = []  # Track which cases have been annotated
    session['start_time'] = datetime.now().isoformat()

    return redirect(url_for('step1'))

@app.route('/overview')
def overview():
    """Overview page showing table of all manipulative cases"""
    case_indices = session.get('case_indices', [])
    dataset_key = session.get('dataset_key', 'mimic')
    model_key = session.get('model_key', 'llama_small')

    # Load agent data
    data = load_data(model_key, dataset_key)

    # Load principal file to get model predictions
    dataset_config = get_dataset_config(dataset_key)
    case_model_name = get_case_file_model_name(model_key)
    analysis_file = f'principal_{case_model_name}.json'
    principal_filepath = os.path.join(dataset_config['cases_dir'], analysis_file)

    principal_data = {}
    if os.path.exists(principal_filepath):
        with open(principal_filepath, 'r') as f:
            principal_json = json.load(f)
            # Create a lookup dict by case_id
            if 'cases' in principal_json:
                for case in principal_json['cases']:
                    principal_data[case['case_id']] = case

    # Build table data
    cases_overview = []
    for position, case_index in enumerate(case_indices):
        case = data[case_index]
        case_id = case.get('case_id')

        # Get model info
        model_info = get_model_info(model_key)
        model_name = model_info['name']

        # Get model prediction from principal data
        is_correct = None
        predicted_answer = None
        correct_answer = case.get('correct_answer_idx') or case.get('correct_answer')

        if case_id in principal_data:
            principal_case = principal_data[case_id]
            # Get the model's prediction for the current model
            model_predictions = principal_case.get('model_predictions', {})
            for model_full_name, prediction in model_predictions.items():
                # Match the model - the keys are like "meta-llama-llama-3.3-70b-instruct"
                if model_key in model_full_name or model_name.lower().replace('.', '').replace('-', '') in model_full_name.lower().replace('.', '').replace('-', ''):
                    is_correct = prediction.get('correct')
                    predicted_answer = prediction.get('predicted_answer')
                    break

        cases_overview.append({
            'position': position,
            'case_id': case_id,
            'case_index': case_index,
            'meta_info': case.get('meta_info', ''),
            'is_correct': is_correct,
            'predicted_answer': predicted_answer,
            'correct_answer': correct_answer,
            'question_preview': case.get('agent_context', '')[:100] + '...' if case.get('agent_context') else ''
        })

    return render_template('overview.html',
                          cases=cases_overview,
                          model_name=model_info['name'],
                          dataset_name=DATASETS[dataset_key]['name'],
                          total_cases=len(cases_overview))

@app.route('/jump_to/<int:position>')
def jump_to(position):
    """Jump to a specific case position"""
    case_indices = session.get('case_indices', [])
    if 0 <= position < len(case_indices):
        session['current_position'] = position
    return redirect(url_for('step1'))

@app.route('/step1')
def step1():
    """Step 1: Show clinical context and collect initial treatment plan (3 blocks) or answer (USMLE)"""
    case_indices = session.get('case_indices', [])
    current_position = session.get('current_position', 0)
    annotated_cases = session.get('annotated_cases', [])
    dataset_key = session.get('dataset_key', 'mimic')
    model_key = session.get('model_key', 'llama_small')
    data = load_data(model_key, dataset_key)

    if current_position >= len(case_indices):
        return redirect(url_for('complete'))

    case_index = case_indices[current_position]
    case = data[case_index].copy()

    # Check if this is USMLE dataset (or USMLE sample)
    if dataset_key in ['usmle', 'usmle_sample']:
        # USMLE: multiple-choice interface
        return render_template('step1_usmle.html',
                              case=case,
                              case_index=case_index,
                              current_position=current_position,
                              total_cases=len(case_indices),
                              annotated_cases=annotated_cases)
    else:
        # MIMIC: short-answer interface
        # Render markdown fields to HTML
        case['agent_context_html'] = render_markdown(case.get('agent_context', ''))

        return render_template('step1.html',
                              case=case,
                              case_index=case_index,
                              current_position=current_position,
                              total_cases=len(case_indices),
                              annotated_cases=annotated_cases)

@app.route('/step1_submit', methods=['POST'])
def step1_submit():
    """Handle step 1 submission - collect 3 treatment blocks with beliefs (MIMIC) or answer choice (USMLE)"""
    dataset_key = session.get('dataset_key', 'mimic')

    if dataset_key in ['usmle', 'usmle_sample']:
        # USMLE: collect answer and belief
        answer = request.form.get('answer')
        answer_belief = request.form.get('answer_belief')

        session['step1_answer'] = answer
        session['step1_answer_belief'] = answer_belief
        session['step1_time'] = datetime.now().isoformat()
    else:
        # MIMIC: collect medications, procedures, diagnoses
        medications = request.form.get('medications')
        medications_belief = request.form.get('medications_belief')

        # Collect procedures
        procedures = request.form.get('procedures')
        procedures_belief = request.form.get('procedures_belief')

        # Collect diagnoses
        diagnoses = request.form.get('diagnoses')
        diagnoses_belief = request.form.get('diagnoses_belief')

        # Store in session
        session['step1_medications'] = medications
        session['step1_medications_belief'] = medications_belief
        session['step1_procedures'] = procedures
        session['step1_procedures_belief'] = procedures_belief
        session['step1_diagnoses'] = diagnoses
        session['step1_diagnoses_belief'] = diagnoses_belief
        session['step1_time'] = datetime.now().isoformat()

    return redirect(url_for('step2'))

@app.route('/step2')
def step2():
    """Step 2: Show agent's recommendation/analysis and allow revision"""
    case_indices = session.get('case_indices', [])
    current_position = session.get('current_position', 0)
    annotated_cases = session.get('annotated_cases', [])
    dataset_key = session.get('dataset_key', 'mimic')
    model_key = session.get('model_key', 'llama_small')
    data = load_data(model_key, dataset_key)

    if current_position >= len(case_indices):
        return redirect(url_for('complete'))

    case_index = case_indices[current_position]
    case = data[case_index].copy()

    # Render markdown fields to HTML
    case['information_html'] = render_markdown(case.get('information', ''))

    if dataset_key in ['usmle', 'usmle_sample']:
        # USMLE: show initial answer and AI's analysis
        initial_answer = session.get('step1_answer', '')
        initial_belief = session.get('step1_answer_belief', '0.5')

        return render_template('step2_usmle.html',
                              case=case,
                              initial_answer=initial_answer,
                              initial_belief=initial_belief,
                              case_index=case_index,
                              current_position=current_position,
                              total_cases=len(case_indices),
                              annotated_cases=annotated_cases)
    else:
        # MIMIC: show initial treatment plan and agent's recommendation
        # Get initial inputs from session (step 1)
        initial_medications = session.get('step1_medications', '')
        initial_medications_belief = session.get('step1_medications_belief', '0.5')
        initial_procedures = session.get('step1_procedures', '')
        initial_procedures_belief = session.get('step1_procedures_belief', '0.5')
        initial_diagnoses = session.get('step1_diagnoses', '')
        initial_diagnoses_belief = session.get('step1_diagnoses_belief', '0.5')

        return render_template('step2.html',
                              case=case,
                              initial_medications=initial_medications,
                              initial_medications_belief=initial_medications_belief,
                              initial_procedures=initial_procedures,
                              initial_procedures_belief=initial_procedures_belief,
                              initial_diagnoses=initial_diagnoses,
                              initial_diagnoses_belief=initial_diagnoses_belief,
                              case_index=case_index,
                              current_position=current_position,
                              total_cases=len(case_indices),
                              annotated_cases=annotated_cases)

@app.route('/step2_submit', methods=['POST'])
def step2_submit():
    """Handle step 2 submission - collect revisions after seeing agent recommendation/analysis"""
    dataset_key = session.get('dataset_key', 'mimic')

    # Get highlights data from step 2
    highlights_data_step2 = request.form.get('highlights_data', '[]')
    try:
        highlights_step2 = json.loads(highlights_data_step2)
    except:
        highlights_step2 = []

    if dataset_key in ['usmle', 'usmle_sample']:
        # USMLE: collect revised answer and belief
        answer_step2 = request.form.get('answer_step2')
        answer_belief_step2 = request.form.get('answer_belief_step2')

        session['step2_answer'] = answer_step2
        session['step2_answer_belief'] = answer_belief_step2
        session['step2_time'] = datetime.now().isoformat()
    else:
        # MIMIC: collect revised medications, procedures, diagnoses
        # Collect revised medications
        medications_step2 = request.form.get('medications_step2')
        medications_belief_step2 = request.form.get('medications_belief_step2')

        # Collect revised procedures
        procedures_step2 = request.form.get('procedures_step2')
        procedures_belief_step2 = request.form.get('procedures_belief_step2')

        # Collect revised diagnoses
        diagnoses_step2 = request.form.get('diagnoses_step2')
        diagnoses_belief_step2 = request.form.get('diagnoses_belief_step2')

        # Store in session
        session['step2_medications'] = medications_step2
        session['step2_medications_belief'] = medications_belief_step2
        session['step2_procedures'] = procedures_step2
        session['step2_procedures_belief'] = procedures_belief_step2
        session['step2_diagnoses'] = diagnoses_step2
        session['step2_diagnoses_belief'] = diagnoses_belief_step2
        session['step2_time'] = datetime.now().isoformat()

    # Store highlights from step 2
    session['step2_highlights'] = highlights_step2

    return redirect(url_for('step3'))

@app.route('/step3')
def step3():
    """Step 3: Show actual treatment/correct answer and allow final revision"""
    case_indices = session.get('case_indices', [])
    current_position = session.get('current_position', 0)
    annotated_cases = session.get('annotated_cases', [])
    dataset_key = session.get('dataset_key', 'mimic')
    model_key = session.get('model_key', 'llama_small')
    data = load_data(model_key, dataset_key)

    if current_position >= len(case_indices):
        return redirect(url_for('complete'))

    case_index = case_indices[current_position]
    case = data[case_index].copy()

    # Render markdown fields to HTML
    case['information_html'] = render_markdown(case.get('information', ''))

    if dataset_key in ['usmle', 'usmle_sample']:
        # USMLE: show step 2 answer and correct answer
        step2_answer = session.get('step2_answer', '')
        step2_belief = session.get('step2_answer_belief', '0.5')

        return render_template('step3_usmle.html',
                              case=case,
                              step2_answer=step2_answer,
                              step2_belief=step2_belief,
                              case_index=case_index,
                              current_position=current_position,
                              total_cases=len(case_indices),
                              annotated_cases=annotated_cases)
    else:
        # MIMIC: show step 2 treatment plan and ground truth
        # Get step 2 inputs from session
        step2_medications = session.get('step2_medications', '')
        step2_medications_belief = session.get('step2_medications_belief', '0.5')
        step2_procedures = session.get('step2_procedures', '')
        step2_procedures_belief = session.get('step2_procedures_belief', '0.5')
        step2_diagnoses = session.get('step2_diagnoses', '')
        step2_diagnoses_belief = session.get('step2_diagnoses_belief', '0.5')

        # Render ground truth fields to HTML
        if 'ground_truth' in case and case['ground_truth']:
            case['ground_truth']['medications_html'] = render_markdown(case['ground_truth'].get('medications', ''))
            case['ground_truth']['procedures_html'] = render_markdown(case['ground_truth'].get('procedures', ''))
            case['ground_truth']['diagnoses_html'] = render_markdown(case['ground_truth'].get('diagnoses', ''))

        return render_template('step3.html',
                              case=case,
                              step2_medications=step2_medications,
                              step2_medications_belief=step2_medications_belief,
                              step2_procedures=step2_procedures,
                              step2_procedures_belief=step2_procedures_belief,
                              step2_diagnoses=step2_diagnoses,
                              step2_diagnoses_belief=step2_diagnoses_belief,
                              case_index=case_index,
                              current_position=current_position,
                              total_cases=len(case_indices),
                              annotated_cases=annotated_cases)

@app.route('/step3_submit', methods=['POST'])
def step3_submit():
    """Handle step 3 submission and save complete annotation"""
    case_indices = session.get('case_indices', [])
    current_position = session.get('current_position', 0)
    dataset_key = session.get('dataset_key', 'mimic')
    model_key = session.get('model_key', 'llama_small')
    data = load_data(model_key, dataset_key)
    case_index = case_indices[current_position]
    case = data[case_index]

    # Get reasoning
    reasoning = request.form.get('reasoning', '')

    # Get highlights data from step 3
    highlights_data_step3 = request.form.get('highlights_data', '[]')
    try:
        highlights_step3 = json.loads(highlights_data_step3)
    except:
        highlights_step3 = []

    # Combine highlights from step 2 and step 3
    highlights_step2 = session.get('step2_highlights', [])

    # Tag highlights with their step
    for h in highlights_step2:
        h['step'] = 'step2'
    for h in highlights_step3:
        h['step'] = 'step3'

    # Combine all highlights
    highlights = highlights_step2 + highlights_step3

    if dataset_key in ['usmle', 'usmle_sample']:
        # USMLE: collect final answer
        answer_step3 = request.form.get('answer_step3')
        answer_belief_step3 = request.form.get('answer_belief_step3')

        # Create comprehensive annotation record for USMLE
        annotation = {
            'annotator_id': session.get('annotator_id'),
            'demographics': session.get('demographics', {}),
            'dataset': dataset_key,
            'model_key': model_key,
            'case_id': case['case_id'],
            'agent_name': case['agent_name'],
            'agent_model': case['agent_model'],
            'correct_answer': case.get('correct_answer'),
            'correct_answer_idx': case.get('correct_answer_idx'),

            # Step 1: Initial answer (before seeing AI's analysis)
            'step1': {
                'answer': session.get('step1_answer', ''),
                'answer_belief': float(session.get('step1_answer_belief', 0.5)),
                'is_correct': session.get('step1_answer') == case.get('correct_answer_idx')
            },

            # Step 2: Revised answer (after seeing AI's analysis)
            'step2': {
                'answer': session.get('step2_answer', ''),
                'answer_belief': float(session.get('step2_answer_belief', 0.5)),
                'is_correct': session.get('step2_answer') == case.get('correct_answer_idx')
            },

            # Step 3: Final answer (after seeing correct answer)
            'step3': {
                'answer': answer_step3,
                'answer_belief': float(answer_belief_step3),
                'is_correct': answer_step3 == case.get('correct_answer_idx')
            },

            # Check if answers changed between steps
            'step1_to_step2_changes': {
                'answer_changed': session.get('step1_answer') != session.get('step2_answer'),
                'answer_belief_changed': session.get('step1_answer_belief') != session.get('step2_answer_belief')
            },

            'step2_to_step3_changes': {
                'answer_changed': session.get('step2_answer') != answer_step3,
                'answer_belief_changed': session.get('step2_answer_belief') != answer_belief_step3
            },

            'reasoning': reasoning,
            'highlights': highlights,
            'session_start': session.get('start_time'),
            'step1_time': session.get('step1_time'),
            'step2_time': session.get('step2_time'),
            'step3_time': datetime.now().isoformat(),
            'timestamp': datetime.now().isoformat()
        }
    else:
        # MIMIC: collect final medications, procedures, diagnoses
        medications_step3 = request.form.get('medications_step3')
        medications_belief_step3 = request.form.get('medications_belief_step3')
        procedures_step3 = request.form.get('procedures_step3')
        procedures_belief_step3 = request.form.get('procedures_belief_step3')
        diagnoses_step3 = request.form.get('diagnoses_step3')
        diagnoses_belief_step3 = request.form.get('diagnoses_belief_step3')

        # Create comprehensive annotation record for MIMIC
        annotation = {
            'annotator_id': session.get('annotator_id'),
            'demographics': session.get('demographics', {}),
            'dataset': dataset_key,
            'model_key': model_key,
            'case_id': case['case_id'],
            'hadm_id': case.get('hadm_id'),
            'subject_id': case.get('subject_id'),
            'agent_name': case['agent_name'],
            'agent_model': case['agent_model'],

            # Step 1: Initial responses (before seeing agent recommendation)
            'step1': {
                'medications': session.get('step1_medications', ''),
                'medications_belief': float(session.get('step1_medications_belief', 0.5)),
                'procedures': session.get('step1_procedures', ''),
                'procedures_belief': float(session.get('step1_procedures_belief', 0.5)),
                'diagnoses': session.get('step1_diagnoses', ''),
                'diagnoses_belief': float(session.get('step1_diagnoses_belief', 0.5))
            },

            # Step 2: Revised responses (after seeing agent recommendation)
            'step2': {
                'medications': session.get('step2_medications', ''),
                'medications_belief': float(session.get('step2_medications_belief', 0.5)),
                'procedures': session.get('step2_procedures', ''),
                'procedures_belief': float(session.get('step2_procedures_belief', 0.5)),
                'diagnoses': session.get('step2_diagnoses', ''),
                'diagnoses_belief': float(session.get('step2_diagnoses_belief', 0.5))
            },

            # Step 3: Final responses (after seeing actual treatment)
            'step3': {
                'medications': medications_step3,
                'medications_belief': float(medications_belief_step3),
                'procedures': procedures_step3,
                'procedures_belief': float(procedures_belief_step3),
                'diagnoses': diagnoses_step3,
                'diagnoses_belief': float(diagnoses_belief_step3)
            },

            # Check if responses changed between steps
            'step1_to_step2_changes': {
                'medications_changed': session.get('step1_medications') != session.get('step2_medications'),
                'medications_belief_changed': session.get('step1_medications_belief') != session.get('step2_medications_belief'),
                'procedures_changed': session.get('step1_procedures') != session.get('step2_procedures'),
                'procedures_belief_changed': session.get('step1_procedures_belief') != session.get('step2_procedures_belief'),
                'diagnoses_changed': session.get('step1_diagnoses') != session.get('step2_diagnoses'),
                'diagnoses_belief_changed': session.get('step1_diagnoses_belief') != session.get('step2_diagnoses_belief')
            },

            'step2_to_step3_changes': {
                'medications_changed': session.get('step2_medications') != medications_step3,
                'medications_belief_changed': session.get('step2_medications_belief') != medications_belief_step3,
                'procedures_changed': session.get('step2_procedures') != procedures_step3,
                'procedures_belief_changed': session.get('step2_procedures_belief') != procedures_belief_step3,
                'diagnoses_changed': session.get('step2_diagnoses') != diagnoses_step3,
                'diagnoses_belief_changed': session.get('step2_diagnoses_belief') != diagnoses_belief_step3
            },

            'reasoning': reasoning,
            'highlights': highlights,
            'session_start': session.get('start_time'),
            'step1_time': session.get('step1_time'),
            'step2_time': session.get('step2_time'),
            'step3_time': datetime.now().isoformat(),
            'timestamp': datetime.now().isoformat()
        }

    # Save annotation
    filepath = save_annotation(annotation, dataset_key)
    print(f"Saved annotation to {filepath}")

    # Mark this case as annotated
    annotated_cases = session.get('annotated_cases', [])
    if current_position not in annotated_cases:
        annotated_cases.append(current_position)
        session['annotated_cases'] = annotated_cases

    # Check if user wants to move to next case or stay
    action = request.form.get('action', 'next')

    if action == 'next':
        # Move to next unannotated case, or next case if all are annotated
        next_position = current_position + 1
        while next_position < len(case_indices) and next_position in annotated_cases:
            next_position += 1

        if next_position < len(case_indices):
            session['current_position'] = next_position
            return redirect(url_for('step1'))
        else:
            return redirect(url_for('complete'))
    else:
        # Stay on current page or go back to step1
        return redirect(url_for('step1'))

@app.route('/summary')
def summary():
    """Summary page showing annotation progress and statistics"""
    annotator_id = session.get('annotator_id', 'anonymous')
    case_indices = session.get('case_indices', [])
    annotated_cases = session.get('annotated_cases', [])
    dataset_key = session.get('dataset_key', 'mimic')
    model_key = session.get('model_key', 'llama_small')

    # Calculate statistics
    total_cases = len(case_indices)
    completed_cases = len(annotated_cases)
    remaining_cases = total_cases - completed_cases

    # Calculate how many times the participant changed their decisions
    # We'll need to check the annotation files for this
    dataset_config = get_dataset_config(dataset_key)
    annotation_dir = dataset_config['annotation_dir']

    decision_changes_count = 0
    if os.path.exists(annotation_dir):
        # Get all annotation files for this annotator
        for filename in os.listdir(annotation_dir):
            if filename.endswith('.json') and annotator_id in filename:
                try:
                    filepath = os.path.join(annotation_dir, filename)
                    with open(filepath, 'r') as f:
                        annotation = json.load(f)

                    # Check if this annotation is from the current session
                    if annotation.get('annotator_id') == annotator_id:
                        # Count decision changes based on dataset type
                        if dataset_key in ['usmle', 'usmle_sample']:
                            # USMLE: check if answer changed between steps
                            if annotation.get('step1_to_step2_changes', {}).get('answer_changed'):
                                decision_changes_count += 1
                            elif annotation.get('step2_to_step3_changes', {}).get('answer_changed'):
                                decision_changes_count += 1
                        else:
                            # MIMIC: check if any treatment component changed
                            step1_to_step2 = annotation.get('step1_to_step2_changes', {})
                            step2_to_step3 = annotation.get('step2_to_step3_changes', {})

                            if (step1_to_step2.get('medications_changed') or
                                step1_to_step2.get('procedures_changed') or
                                step1_to_step2.get('diagnoses_changed')):
                                decision_changes_count += 1
                            elif (step2_to_step3.get('medications_changed') or
                                  step2_to_step3.get('procedures_changed') or
                                  step2_to_step3.get('diagnoses_changed')):
                                decision_changes_count += 1
                except Exception as e:
                    print(f"Error reading annotation file {filename}: {e}")
                    continue

    # Generate Prolific authentication code
    # Use a hash of annotator_id, timestamp, and completed cases to create unique code
    session_start = session.get('start_time', datetime.now().isoformat())
    code_string = f"{annotator_id}_{session_start}_{completed_cases}_{total_cases}"
    prolific_code = hashlib.sha256(code_string.encode()).hexdigest()[:12].upper()

    # Load cases overview data (for manipulative cases)
    cases_overview = []
    show_cases_table = False

    # Check if this is a manipulative cases session
    if case_indices:
        try:
            # Load agent data
            data = load_data(model_key, dataset_key)

            # Load principal file to get model predictions
            case_model_name = get_case_file_model_name(model_key)
            analysis_file = f'principal_{case_model_name}.json'
            principal_filepath = os.path.join(dataset_config['cases_dir'], analysis_file)

            principal_data = {}
            if os.path.exists(principal_filepath):
                show_cases_table = True
                with open(principal_filepath, 'r') as f:
                    principal_json = json.load(f)
                    # Create a lookup dict by case_id
                    if 'cases' in principal_json:
                        for case in principal_json['cases']:
                            principal_data[case['case_id']] = case

                # Build table data
                for position, case_index in enumerate(case_indices):
                    case = data[case_index]
                    case_id = case.get('case_id')

                    # Get model info
                    model_info = get_model_info(model_key)
                    model_name = model_info['name']

                    # Get model prediction from principal data
                    is_correct = None
                    predicted_answer = None
                    correct_answer = case.get('correct_answer_idx') or case.get('correct_answer')

                    if case_id in principal_data:
                        principal_case = principal_data[case_id]
                        # Get the model's prediction for the current model
                        model_predictions = principal_case.get('model_predictions', {})
                        for model_full_name, prediction in model_predictions.items():
                            # Match the model - the keys are like "meta-llama-llama-3.3-70b-instruct"
                            if model_key in model_full_name or model_name.lower().replace('.', '').replace('-', '') in model_full_name.lower().replace('.', '').replace('-', ''):
                                is_correct = prediction.get('correct')
                                predicted_answer = prediction.get('predicted_answer')
                                break

                    cases_overview.append({
                        'position': position,
                        'case_id': case_id,
                        'case_index': case_index,
                        'meta_info': case.get('meta_info', ''),
                        'is_correct': is_correct,
                        'predicted_answer': predicted_answer,
                        'correct_answer': correct_answer,
                        'question_preview': case.get('agent_context', '')[:100] + '...' if case.get('agent_context') else ''
                    })
        except Exception as e:
            print(f"Error loading cases overview: {e}")
            show_cases_table = False

    return render_template('summary.html',
                          annotator_id=annotator_id,
                          total_cases=total_cases,
                          completed_cases=completed_cases,
                          remaining_cases=remaining_cases,
                          decision_changes_count=decision_changes_count,
                          prolific_code=prolific_code,
                          dataset_name=DATASETS[dataset_key]['name'],
                          model_name=get_model_info(model_key)['name'],
                          cases_overview=cases_overview,
                          show_cases_table=show_cases_table)

@app.route('/complete')
def complete():
    """Completion page - redirects to summary"""
    return redirect(url_for('summary'))

@app.route('/revoke_consent')
def revoke_consent():
    """Revoke consent and clear consent-related session data"""
    session['consent_given'] = False
    if 'consent_timestamp' in session:
        del session['consent_timestamp']
    return redirect(url_for('consent'))

@app.route('/reset')
def reset():
    """Reset session"""
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
