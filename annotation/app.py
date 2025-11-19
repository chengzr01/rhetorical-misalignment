from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import json
import os
from datetime import datetime
import markdown

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

# Dataset configuration
DATASETS = {
    'mimic': {
        'name': 'MIMIC-IV',
        'data_dir': '../experiments/cache/mimiciv_demo',
        'cases_dir': '../experiments/cases/mimiciv_demo',
        'annotation_dir': 'annotations/mimic'
    },
    'usmle': {
        'name': 'USMLE',
        'data_dir': '../experiments/cache/usmle',
        'cases_dir': '../experiments/cases/usmle',
        'annotation_dir': 'annotations/usmle'
    }
}

# Ensure annotation directories exist
for dataset_key, dataset_config in DATASETS.items():
    os.makedirs(dataset_config['annotation_dir'], exist_ok=True)

# Available model files (in order as per README.md)
AVAILABLE_MODELS = [
    {'key': 'small_dpo', 'file': 'agent_llama-dpo.json', 'name': 'Llama-3.1-8B-Instruct-Tulu-3-DPO'},
    {'key': 'small_sft', 'file': 'agent_llama-sft.json', 'name': 'Llama-3.1-8B-Instruct-Tulu-3-SFT'},
    {'key': 'llama', 'file': 'agent_llama.json', 'name': 'Llama-3.3-70B-Instruct'},
    {'key': 'llama_large', 'file': 'agent_llama-large.json', 'name': 'Llama-3.1-405B-Instruct'},
    {'key': 'deepseek', 'file': 'agent_deepseek.json', 'name': 'DeepSeek-V3.1'},
]

def get_dataset_config(dataset_key):
    """Get dataset configuration"""
    return DATASETS.get(dataset_key, DATASETS['mimic'])

def get_model_info(model_key):
    """Get model info by key"""
    for model in AVAILABLE_MODELS:
        if model['key'] == model_key:
            return model
    return AVAILABLE_MODELS[0]  # Default to first model

def get_available_models(dataset_key='mimic'):
    """Get list of available model files in order for given dataset"""
    dataset_config = get_dataset_config(dataset_key)
    available = []
    for model in AVAILABLE_MODELS:
        filepath = os.path.join(dataset_config['data_dir'], model['file'])
        model_copy = model.copy()
        model_copy['available'] = os.path.exists(filepath)
        available.append(model_copy)
    return available

def check_dataset_availability():
    """Check which datasets have available data"""
    dataset_availability = {}
    for dataset_key, dataset_config in DATASETS.items():
        # Check if data directory exists and has at least one model file
        data_dir = dataset_config['data_dir']
        has_data = False
        if os.path.exists(data_dir):
            for model in AVAILABLE_MODELS:
                filepath = os.path.join(data_dir, model['file'])
                if os.path.exists(filepath):
                    has_data = True
                    break
        dataset_availability[dataset_key] = has_data
    return dataset_availability

# Configure markdown converter
md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists'])

def load_data(model_key='small_dpo', dataset_key='mimic'):
    """Load data for specified model and dataset"""
    dataset_config = get_dataset_config(dataset_key)
    model_info = get_model_info(model_key)
    filepath = os.path.join(dataset_config['data_dir'], model_info['file'])
    with open(filepath, 'r') as f:
        return json.load(f)

def render_markdown(text):
    """Convert markdown text to HTML"""
    if not text:
        return ""
    # Reset markdown converter for each use
    md.reset()
    return md.convert(text)

def save_annotation(annotation, dataset_key='mimic'):
    """Save annotation to a file in dataset-specific directory"""
    dataset_config = get_dataset_config(dataset_key)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{annotation['case_id']}_{annotation['annotator_id']}_{timestamp}.json"
    filepath = os.path.join(dataset_config['annotation_dir'], filename)

    with open(filepath, 'w') as f:
        json.dump(annotation, f, indent=2)

    return filepath

def parse_indices(indices_string, total_cases):
    """Parse a string of indices into a list of integers

    Supports formats:
    - Comma-separated: "0,5,10,15"
    - Range: "0-20"
    - Mixed: "0,5,10-15,20"
    """
    indices = []

    if not indices_string or not indices_string.strip():
        return []

    parts = indices_string.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range format
            try:
                start, end = part.split('-')
                start = int(start.strip())
                end = int(end.strip())
                indices.extend(range(start, end + 1))
            except ValueError:
                continue
        else:
            # Single index
            try:
                indices.append(int(part))
            except ValueError:
                continue

    # Filter to valid indices
    indices = [i for i in indices if 0 <= i < total_cases]
    # Remove duplicates and sort
    indices = sorted(set(indices))

    return indices

def get_random_indices(total_cases, count):
    """Get random case indices"""
    import random
    count = min(count, total_cases)
    indices = list(range(total_cases))
    random.shuffle(indices)
    return sorted(indices[:count])

def load_manipulative_case_ids(model_key, dataset_key='mimic'):
    """Load manipulative case IDs from decision making analysis file"""
    dataset_config = get_dataset_config(dataset_key)
    # Map model key to analysis file
    analysis_file = f'decision_making_analysis_{model_key}_max_diff.json'
    filepath = os.path.join(dataset_config['cases_dir'], analysis_file)

    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        case_ids = set()

        # Check if this is MIMIC format (has 'principals') or USMLE format (has 'cases')
        if 'principals' in data:
            # MIMIC format: Extract all case_ids from all principals
            principals = data.get('principals', {})
            for principal_name, cases in principals.items():
                for case in cases:
                    case_ids.add(case['case_id'])
        elif 'cases' in data:
            # USMLE format: Extract case_ids from the cases array
            cases = data.get('cases', [])
            for case in cases:
                case_ids.add(case['case_id'])
        else:
            print(f"Unknown format in {filepath}")
            return []

        return list(case_ids)
    except Exception as e:
        print(f"Error loading manipulative cases: {e}")
        return []

def get_indices_for_case_ids(data, case_ids):
    """Find indices in data array that match the given case_ids"""
    indices = []
    for idx, case in enumerate(data):
        if case.get('case_id') in case_ids:
            indices.append(idx)
    return sorted(indices)

@app.route('/')
def index():
    """Landing page - show consent form"""
    return render_template('index.html')

@app.route('/demographics')
def demographics():
    """Demographics and study information page"""
    # Check dataset availability
    dataset_availability = check_dataset_availability()

    # Find first available dataset
    default_dataset = 'usmle'
    for dataset_key in ['usmle', 'mimic']:
        if dataset_availability.get(dataset_key, False):
            default_dataset = dataset_key
            break

    # Get available models for default dataset
    available_models = get_available_models(default_dataset)

    # Get default model to show total cases
    available_model_keys = [m['key'] for m in available_models if m.get('available', False)]
    default_model = available_model_keys[0] if available_model_keys else 'small_dpo'

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
                          dataset_availability=dataset_availability)

@app.route('/api/models/<dataset_key>')
def api_get_models(dataset_key):
    """API endpoint to get available models for a dataset"""
    models = get_available_models(dataset_key)
    return jsonify({'models': models})

@app.route('/start', methods=['POST'])
def start_annotation():
    """Start annotation session"""
    annotator_id = request.form.get('annotator_id', 'anonymous')
    dataset_key = request.form.get('dataset', 'mimic')
    selection_mode = request.form.get('selection_mode', 'all')
    model_key = request.form.get('model_key', 'small_dpo')

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
            case_indices = get_indices_for_case_ids(data, manipulative_case_ids)
            if not case_indices:
                # Fallback to all cases if no manipulative cases found
                case_indices = list(range(total_cases))
        else:
            # Random sampling
            random_count = int(request.form.get('random_count', 10))
            case_indices = get_random_indices(total_cases, random_count)
    else:
        # All cases in order
        case_index = int(request.form.get('case_index', 0))
        case_indices = list(range(case_index, total_cases))

    session['annotator_id'] = annotator_id if annotator_id else 'anonymous'
    session['demographics'] = demographics
    session['dataset_key'] = dataset_key
    session['model_key'] = model_key
    session['case_indices'] = case_indices
    session['current_position'] = 0
    session['annotated_cases'] = []  # Track which cases have been annotated
    session['start_time'] = datetime.now().isoformat()

    return redirect(url_for('step1'))

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
    model_key = session.get('model_key', 'small_dpo')
    data = load_data(model_key, dataset_key)

    if current_position >= len(case_indices):
        return redirect(url_for('complete'))

    case_index = case_indices[current_position]
    case = data[case_index].copy()

    # Check if this is USMLE dataset
    if dataset_key == 'usmle':
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

    if dataset_key == 'usmle':
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
    model_key = session.get('model_key', 'small_dpo')
    data = load_data(model_key, dataset_key)

    if current_position >= len(case_indices):
        return redirect(url_for('complete'))

    case_index = case_indices[current_position]
    case = data[case_index].copy()

    # Render markdown fields to HTML
    case['information_html'] = render_markdown(case.get('information', ''))

    if dataset_key == 'usmle':
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

    if dataset_key == 'usmle':
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
    model_key = session.get('model_key', 'small_dpo')
    data = load_data(model_key, dataset_key)

    if current_position >= len(case_indices):
        return redirect(url_for('complete'))

    case_index = case_indices[current_position]
    case = data[case_index].copy()

    # Render markdown fields to HTML
    case['information_html'] = render_markdown(case.get('information', ''))

    if dataset_key == 'usmle':
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
    model_key = session.get('model_key', 'small_dpo')
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

    if dataset_key == 'usmle':
        # USMLE: collect final answer
        answer_step3 = request.form.get('answer_step3')
        answer_belief_step3 = request.form.get('answer_belief_step3')

        # Create comprehensive annotation record for USMLE
        annotation = {
            'annotator_id': session.get('annotator_id'),
            'demographics': session.get('demographics', {}),
            'dataset': dataset_key,
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
    import hashlib

    annotator_id = session.get('annotator_id', 'anonymous')
    case_indices = session.get('case_indices', [])
    annotated_cases = session.get('annotated_cases', [])
    dataset_key = session.get('dataset_key', 'mimic')
    model_key = session.get('model_key', 'small_dpo')

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
                        if dataset_key == 'usmle':
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

    return render_template('summary.html',
                          annotator_id=annotator_id,
                          total_cases=total_cases,
                          completed_cases=completed_cases,
                          remaining_cases=remaining_cases,
                          decision_changes_count=decision_changes_count,
                          prolific_code=prolific_code,
                          dataset_name=DATASETS[dataset_key]['name'])

@app.route('/complete')
def complete():
    """Completion page - redirects to summary"""
    return redirect(url_for('summary'))

@app.route('/reset')
def reset():
    """Reset session"""
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
