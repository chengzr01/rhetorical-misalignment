from flask import Flask, render_template, request, redirect, url_for, session
import json
import os
from datetime import datetime
import markdown

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

# Load the data
DATA_DIR = '../experiments/cache'
CASES_DIR = '../experiments/cases'
ANNOTATION_DIR = 'annotations'

# Ensure annotation directory exists
os.makedirs(ANNOTATION_DIR, exist_ok=True)

# Available model files (in order as per README.md)
AVAILABLE_MODELS = [
    {'key': 'small_dpo', 'file': 'agent_small_dpo.json', 'name': 'Llama-3.1-8B-Instruct-Tulu-3-DPO'},
    {'key': 'small_sft', 'file': 'agent_small_sft.json', 'name': 'Llama-3.1-8B-Instruct-Tulu-3-SFT'},
    {'key': 'llama', 'file': 'agent_llama.json', 'name': 'Llama-3.3-70B-Instruct'},
    {'key': 'oss', 'file': 'agent_oss.json', 'name': 'GPT-OSS-120B'},
    {'key': 'deepseek', 'file': 'agent_deepseek.json', 'name': 'DeepSeek-V3.1'},
]

def get_model_info(model_key):
    """Get model info by key"""
    for model in AVAILABLE_MODELS:
        if model['key'] == model_key:
            return model
    return AVAILABLE_MODELS[0]  # Default to first model

def get_available_models():
    """Get list of available model files in order"""
    available = []
    for model in AVAILABLE_MODELS:
        filepath = os.path.join(DATA_DIR, model['file'])
        if os.path.exists(filepath):
            available.append(model)
    return available

# Configure markdown converter
md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists'])

def load_data(model_key='small_dpo'):
    """Load data for specified model"""
    model_info = get_model_info(model_key)
    filepath = os.path.join(DATA_DIR, model_info['file'])
    with open(filepath, 'r') as f:
        return json.load(f)

def render_markdown(text):
    """Convert markdown text to HTML"""
    if not text:
        return ""
    # Reset markdown converter for each use
    md.reset()
    return md.convert(text)

def save_annotation(annotation):
    """Save annotation to a file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{annotation['case_id']}_{annotation['annotator_id']}_{timestamp}.json"
    filepath = os.path.join(ANNOTATION_DIR, filename)

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

def load_manipulative_case_ids(model_key):
    """Load manipulative case IDs from decision making analysis file"""
    # Map model key to analysis file
    analysis_file = f'decision_making_analysis_{model_key}_max_diff.json'
    filepath = os.path.join(CASES_DIR, analysis_file)

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
    available_models = get_available_models()
    # Get default model to show total cases
    default_model = available_models[0]['key'] if available_models else 'small_dpo'
    data = load_data(default_model)
    return render_template('index.html',
                          total_cases=len(data),
                          available_models=available_models)

@app.route('/start', methods=['POST'])
def start_annotation():
    """Start annotation session"""
    annotator_id = request.form.get('annotator_id', 'anonymous')
    selection_mode = request.form.get('selection_mode', 'all')
    model_key = request.form.get('model_key', 'small_dpo')

    # Load data for selected model
    data = load_data(model_key)
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
            # Load manipulative cases for this model
            manipulative_case_ids = load_manipulative_case_ids(model_key)
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
    """Step 1: Show principal context and agent recommendation"""
    case_indices = session.get('case_indices', [])
    current_position = session.get('current_position', 0)
    annotated_cases = session.get('annotated_cases', [])
    model_key = session.get('model_key', 'small_dpo')
    data = load_data(model_key)

    if current_position >= len(case_indices):
        return redirect(url_for('complete'))

    case_index = case_indices[current_position]
    case = data[case_index].copy()

    # Render markdown fields to HTML
    case['principal_context_html'] = render_markdown(case.get('principal_context', ''))
    case['information_html'] = render_markdown(case.get('information', ''))

    return render_template('step1.html',
                          case=case,
                          case_index=case_index,
                          current_position=current_position,
                          total_cases=len(case_indices),
                          annotated_cases=annotated_cases)

@app.route('/step1_submit', methods=['POST'])
def step1_submit():
    """Handle step 1 submission"""
    decision = request.form.get('decision')

    session['initial_decision'] = decision
    session['step1_time'] = datetime.now().isoformat()

    return redirect(url_for('step2'))

@app.route('/step2')
def step2():
    """Step 2: Show agent context, ground truth, and ask for revision"""
    case_indices = session.get('case_indices', [])
    current_position = session.get('current_position', 0)
    annotated_cases = session.get('annotated_cases', [])
    model_key = session.get('model_key', 'small_dpo')
    data = load_data(model_key)

    if current_position >= len(case_indices):
        return redirect(url_for('complete'))

    case_index = case_indices[current_position]
    case = data[case_index].copy()
    initial_decision = session.get('initial_decision')

    # Render markdown fields to HTML
    case['agent_context_html'] = render_markdown(case.get('agent_context', ''))
    case['information_html'] = render_markdown(case.get('information', ''))

    # Render ground truth fields to HTML
    if 'ground_truth' in case and case['ground_truth']:
        case['ground_truth']['medications_html'] = render_markdown(case['ground_truth'].get('medications', ''))
        case['ground_truth']['procedures_html'] = render_markdown(case['ground_truth'].get('procedures', ''))
        case['ground_truth']['diagnoses_html'] = render_markdown(case['ground_truth'].get('diagnoses', ''))

    return render_template('step2.html',
                          case=case,
                          initial_decision=initial_decision,
                          case_index=case_index,
                          current_position=current_position,
                          total_cases=len(case_indices),
                          annotated_cases=annotated_cases)

@app.route('/step2_submit', methods=['POST'])
def step2_submit():
    """Handle step 2 submission and save annotation"""
    final_decision = request.form.get('decision')
    reasoning = request.form.get('reasoning', '')

    case_indices = session.get('case_indices', [])
    current_position = session.get('current_position', 0)
    model_key = session.get('model_key', 'small_dpo')
    data = load_data(model_key)
    case_index = case_indices[current_position]
    case = data[case_index]

    # Create annotation record
    annotation = {
        'annotator_id': session.get('annotator_id'),
        'case_id': case['case_id'],
        'hadm_id': case['hadm_id'],
        'subject_id': case['subject_id'],
        'agent_name': case['agent_name'],
        'agent_model': case['agent_model'],
        'initial_decision': session.get('initial_decision'),
        'final_decision': final_decision,
        'decision_changed': session.get('initial_decision') != final_decision,
        'reasoning': reasoning,
        'session_start': session.get('start_time'),
        'step1_time': session.get('step1_time'),
        'step2_time': datetime.now().isoformat(),
        'timestamp': datetime.now().isoformat()
    }

    # Save annotation
    filepath = save_annotation(annotation)
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
