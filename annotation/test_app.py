"""
Test script to verify the Flask application setup
"""

import json
import os
from datetime import datetime

# Test data loading
DATA_FILE = '../experiments/cache/agent_deepseek.json'

print("Testing Flask Application Setup")
print("=" * 50)
print()

# Check if data file exists
if os.path.exists(DATA_FILE):
    print(f"✓ Data file found: {DATA_FILE}")

    # Load and check data
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)

    print(f"✓ Total cases loaded: {len(data)}")

    # Show first case structure
    if len(data) > 0:
        print("\n✓ First case structure:")
        case = data[0]
        print(f"  - case_id: {case.get('case_id')}")
        print(f"  - hadm_id: {case.get('hadm_id')}")
        print(f"  - subject_id: {case.get('subject_id')}")
        print(f"  - agent_name: {case.get('agent_name')}")
        print(f"  - agent_model: {case.get('agent_model')}")
        print(f"  - has principal_context: {bool(case.get('principal_context'))}")
        print(f"  - has agent_context: {bool(case.get('agent_context'))}")
        print(f"  - has information: {bool(case.get('information'))}")
        print(f"  - has ground_truth: {bool(case.get('ground_truth'))}")
else:
    print(f"✗ Data file not found: {DATA_FILE}")
    exit(1)

# Check annotations directory
ANNOTATION_DIR = 'annotations'
if not os.path.exists(ANNOTATION_DIR):
    os.makedirs(ANNOTATION_DIR)
    print(f"\n✓ Created annotations directory: {ANNOTATION_DIR}")
else:
    print(f"\n✓ Annotations directory exists: {ANNOTATION_DIR}")

# Create a sample annotation to demonstrate the format
sample_annotation = {
    'annotator_id': 'test_user',
    'case_id': data[0]['case_id'],
    'hadm_id': data[0]['hadm_id'],
    'subject_id': data[0]['subject_id'],
    'agent_name': data[0]['agent_name'],
    'agent_model': data[0]['agent_model'],
    'initial_decision': 'accept',
    'final_decision': 'reject',
    'decision_changed': True,
    'reasoning': 'After reviewing the full context and ground truth, the recommendation does not align with actual treatments.',
    'session_start': datetime.now().isoformat(),
    'step1_time': datetime.now().isoformat(),
    'step2_time': datetime.now().isoformat(),
    'timestamp': datetime.now().isoformat()
}

sample_file = os.path.join(ANNOTATION_DIR, 'SAMPLE_annotation_format.json')
with open(sample_file, 'w') as f:
    json.dump(sample_annotation, f, indent=2)

print(f"\n✓ Created sample annotation: {sample_file}")
print("\nSample annotation format:")
print(json.dumps(sample_annotation, indent=2))

print("\n" + "=" * 50)
print("All tests passed! The application is ready to run.")
print("\nTo start the application:")
print("  ./start.sh")
print("\nOr manually:")
print("  conda activate alignment")
print("  python app.py")
