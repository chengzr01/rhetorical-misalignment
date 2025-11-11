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
        'data_dir': '../experiments/cache',
        'cases_dir': '../experiments/cases',
        'annotation_dir': 'annotations/mimic'
    },
    'usmle': {
        'name': 'USMLE',
        'data_dir': '../experiments/cache_usmle',
        'cases_dir': '../experiments/cases_usmle',
        'annotation_dir': 'annotations/usmle'
    }
}

# Ensure annotation directories exist
for dataset_key, dataset_config in DATASETS.items():
    os.makedirs(dataset_config['annotation_dir'], exist_ok=True)

# Available model files (in order as per README.md)
AVAILABLE_MODELS = [
    {'key': 'small_dpo', 'file': 'agent_small_dpo.json', 'name': 'Llama-3.1-8B-Instruct-Tulu-3-DPO'},
    {'key': 'small_sft', 'file': 'agent_small_sft.json', 'name': 'Llama-3.1-8B-Instruct-Tulu-3-SFT'},
    {'key': 'llama', 'file': 'agent_llama.json', 'name': 'Llama-3.3-70B-Instruct'},
    {'key': 'oss', 'file': 'agent_oss.json', 'name': 'GPT-OSS-120B'},
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

        # Extract all case_ids from all principals
        case_ids = set()
        principals = data.get('principals', {})
        for principal_name, cases in principals.items():
            for case in cases:
                case_ids.add(case['case_id'])

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
    """Landing page - ask for annotator ID and show case selection"""
    # Check dataset availability
    dataset_availability = check_dataset_availability()

    # Find first available dataset
    default_dataset = 'mimic'
    for dataset_key in ['mimic', 'usmle']:
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

    return render_template('index.html',
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
    session['dataset_key'] = dataset_key
    session['model_key'] = model_key
    session['case_indices'] = case_indices
    session['current_position'] = 0
    session['annotated_cases'] = []  # Track which cases have been annotated
    session['start_time'] = datetime.now().isoformat()

    return redirect(url_for('demographics'))

@app.route('/demographics')
def demographics():
    """Demographics information collection page"""
    # Check if session has been initialized
    if 'annotator_id' not in session:
        return redirect(url_for('index'))

    return render_template('demographics.html')

@app.route('/demographics_submit', methods=['POST'])
def demographics_submit():
    """Handle demographics submission"""
    # Store demographic information in session
    session['demographics'] = {
        'expertise': request.form.get('expertise'),
        'years_of_practice': request.form.get('years_of_practice'),
        'age': request.form.get('age'),
        'sex': request.form.get('sex'),
        'race': request.form.get('race'),
        'practice_location': request.form.get('practice_location'),
        'submitted_at': datetime.now().isoformat()
    }

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
    """Step 1: Show clinical context and collect initial treatment plan (3 blocks)"""
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
    case['agent_context_html'] = render_markdown(case.get('agent_context', ''))

    return render_template('step1.html',
                          case=case,
                          case_index=case_index,
                          current_position=current_position,
                          total_cases=len(case_indices),
                          annotated_cases=annotated_cases)

@app.route('/step1_submit', methods=['POST'])
def step1_submit():
    """Handle step 1 submission - collect 3 treatment blocks with beliefs"""
    # Collect medications
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
    """Step 2: Show agent's recommendation and allow revision"""
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

    # Get initial inputs from session (step 1)
    initial_medications = session.get('step1_medications', '')
    initial_medications_belief = session.get('step1_medications_belief', '0.5')
    initial_procedures = session.get('step1_procedures', '')
    initial_procedures_belief = session.get('step1_procedures_belief', '0.5')
    initial_diagnoses = session.get('step1_diagnoses', '')
    initial_diagnoses_belief = session.get('step1_diagnoses_belief', '0.5')

    # Render markdown fields to HTML
    case['information_html'] = render_markdown(case.get('information', ''))

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
    """Handle step 2 submission - collect revisions after seeing agent recommendation"""
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

    return redirect(url_for('step3'))

@app.route('/step3')
def step3():
    """Step 3: Show actual treatment and allow final revision"""
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
    # Collect final medications
    medications_step3 = request.form.get('medications_step3')
    medications_belief_step3 = request.form.get('medications_belief_step3')

    # Collect final procedures
    procedures_step3 = request.form.get('procedures_step3')
    procedures_belief_step3 = request.form.get('procedures_belief_step3')

    # Collect final diagnoses
    diagnoses_step3 = request.form.get('diagnoses_step3')
    diagnoses_belief_step3 = request.form.get('diagnoses_belief_step3')

    # Get reasoning
    reasoning = request.form.get('reasoning', '')

    case_indices = session.get('case_indices', [])
    current_position = session.get('current_position', 0)
    dataset_key = session.get('dataset_key', 'mimic')
    model_key = session.get('model_key', 'small_dpo')
    data = load_data(model_key, dataset_key)
    case_index = case_indices[current_position]
    case = data[case_index]

    # Create comprehensive annotation record
    annotation = {
        'annotator_id': session.get('annotator_id'),
        'demographics': session.get('demographics', {}),
        'dataset': dataset_key,
        'case_id': case['case_id'],
        'hadm_id': case['hadm_id'],
        'subject_id': case['subject_id'],
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

@app.route('/complete')
def complete():
    
    """Completion page"""
    annotator_id = session.get('annotator_id', 'anonymous')
    return render_template('complete.html', annotator_id=annotator_id)

@app.route('/reset')
def reset():
    """Reset session"""
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
