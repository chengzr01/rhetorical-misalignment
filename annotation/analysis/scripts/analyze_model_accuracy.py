#!/usr/bin/env python3
"""
Analyze model accuracy on USMLE Sample questions.
Provides detailed accuracy metrics for each model across different question types.
"""

import json
import os
from collections import defaultdict
import statistics


# Test results directory
TESTS_DIR = '../../../experiments/tests'

# Model configurations
MODELS = [
    {
        'key': 'llama_small',
        'name': 'Llama-3.1-8B',
        'file': 'test_usmle_sample_meta-llama-llama-3.1-8b-instruct_belief.json'
    },
    {
        'key': 'llama',
        'name': 'Llama-3.3-70B',
        'file': 'test_usmle_sample_meta-llama-llama-3.3-70b-instruct_belief.json'
    },
    {
        'key': 'llama_large',
        'name': 'Llama-3.1-405B',
        'file': 'test_usmle_sample_meta-llama-llama-3.1-405b-instruct_belief.json'
    },
    {
        'key': 'llama_dpo',
        'name': 'Llama-3.1-Tulu-3-8B-DPO',
        'file': 'test_usmle_sample_allenai-Llama-3.1-Tulu-3-8B-DPO_belief.json'
    },
    {
        'key': 'llama_sft',
        'name': 'Llama-3.1-Tulu-3-8B-SFT',
        'file': 'test_usmle_sample_allenai-Llama-3.1-Tulu-3-8B-SFT_belief.json'
    },
    {
        'key': 'deepseek',
        'name': 'DeepSeek-V3',
        'file': 'test_usmle_sample_deepseek-deepseek-chat-v3.1_belief.json'
    },
]


def calculate_metrics(results):
    """Calculate metrics from results if not present"""
    metrics = {
        'total': len(results),
        'correct': 0,
        'incorrect': 0,
        'no_answer': 0,
    }

    by_step = defaultdict(lambda: {'total': 0, 'correct': 0})
    beliefs_all = []
    beliefs_correct = []
    beliefs_incorrect = []

    for result in results:
        # Count correctness
        if result.get('correct') is True:
            metrics['correct'] += 1
        elif result.get('correct') is False:
            metrics['incorrect'] += 1
        else:
            metrics['no_answer'] += 1

        # Track by step
        step = result.get('meta_info', 'unknown')
        by_step[step]['total'] += 1
        if result.get('correct'):
            by_step[step]['correct'] += 1

        # Track beliefs
        belief = result.get('belief')
        if belief is not None:
            beliefs_all.append(belief)
            if result.get('correct'):
                beliefs_correct.append(belief)
            else:
                beliefs_incorrect.append(belief)

    # Calculate accuracies
    metrics['accuracy'] = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0

    # Add step metrics
    for step_name, step_data in by_step.items():
        metrics[step_name] = {
            'total': step_data['total'],
            'correct': step_data['correct'],
            'accuracy': step_data['correct'] / step_data['total'] if step_data['total'] > 0 else 0
        }

    # Add belief metrics
    metrics['belief'] = {
        'total_with_belief': len(beliefs_all),
        'no_belief': metrics['total'] - len(beliefs_all),
        'mean_belief': statistics.mean(beliefs_all) if beliefs_all else 0,
        'mean_belief_correct': statistics.mean(beliefs_correct) if beliefs_correct else 0,
        'mean_belief_incorrect': statistics.mean(beliefs_incorrect) if beliefs_incorrect else 0,
    }

    return metrics


def load_test_results(filepath):
    """Load test results from JSON file"""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # If metrics field is missing, calculate it
        if 'metrics' not in data and 'results' in data:
            data['metrics'] = calculate_metrics(data['results'])

        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def analyze_by_question_step(results):
    """Analyze accuracy by USMLE step (step1, step2, step3)"""
    by_step = defaultdict(lambda: {'correct': 0, 'total': 0, 'beliefs': []})

    for result in results:
        step = result.get('meta_info', 'unknown')
        correct = result.get('correct', False)
        belief = result.get('belief')

        by_step[step]['total'] += 1
        if correct:
            by_step[step]['correct'] += 1
        if belief is not None:
            by_step[step]['beliefs'].append(belief)

    return by_step


def calculate_calibration(results):
    """Calculate calibration metrics: how well belief matches accuracy"""
    # Group by belief bins
    bins = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    calibration_data = defaultdict(lambda: {'correct': 0, 'total': 0, 'beliefs': []})

    for result in results:
        belief = result.get('belief')
        correct = result.get('correct', False)

        if belief is None:
            continue

        # Find appropriate bin
        bin_label = None
        for i in range(len(bins) - 1):
            if bins[i] <= belief <= bins[i+1]:
                bin_label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                break

        if bin_label:
            calibration_data[bin_label]['total'] += 1
            if correct:
                calibration_data[bin_label]['correct'] += 1
            calibration_data[bin_label]['beliefs'].append(belief)

    return calibration_data


def print_model_analysis(model_info, data):
    """Print detailed analysis for a single model"""
    print(f"\n{'='*100}")
    print(f"MODEL: {model_info['name']} ({model_info['key']})")
    print(f"{'='*100}")

    metrics = data['metrics']
    results = data['results']

    # Overall accuracy
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Total questions: {metrics['total']}")
    print(f"  Correct: {metrics['correct']} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Incorrect: {metrics['incorrect']} ({metrics['incorrect']/metrics['total']*100:.2f}%)")
    if metrics.get('no_answer', 0) > 0:
        print(f"  No answer: {metrics['no_answer']}")

    # Accuracy by USMLE step
    print(f"\n{'-'*100}")
    print(f"ACCURACY BY USMLE STEP:")
    print(f"{'-'*100}")
    print(f"{'Step':<15} {'Total':>8} {'Correct':>10} {'Accuracy':>12} {'% of Total':>12}")
    print(f"{'-'*100}")

    for step_name in ['step1', 'step2', 'step3']:
        if step_name in metrics:
            step_data = metrics[step_name]
            pct_of_total = step_data['total'] / metrics['total'] * 100
            print(f"{step_name:<15} {step_data['total']:>8} {step_data['correct']:>10} "
                  f"{step_data['accuracy']*100:>11.2f}% {pct_of_total:>11.1f}%")

    # Belief/confidence analysis
    belief_metrics = metrics.get('belief', {})
    if belief_metrics:
        print(f"\n{'-'*100}")
        print(f"CONFIDENCE/BELIEF ANALYSIS:")
        print(f"{'-'*100}")
        print(f"  Mean confidence (all): {belief_metrics.get('mean_belief', 0):.3f}")
        print(f"  Mean confidence (correct answers): {belief_metrics.get('mean_belief_correct', 0):.3f}")
        print(f"  Mean confidence (incorrect answers): {belief_metrics.get('mean_belief_incorrect', 0):.3f}")

        diff = belief_metrics.get('mean_belief_correct', 0) - belief_metrics.get('mean_belief_incorrect', 0)
        print(f"  Confidence gap (correct - incorrect): {diff:+.3f}")

        # Calibration: expected accuracy based on confidence
        mean_belief = belief_metrics.get('mean_belief', 0)
        actual_accuracy = metrics['accuracy']
        calibration_error = abs(mean_belief - actual_accuracy)
        print(f"\n  Expected accuracy (based on mean confidence): {mean_belief*100:.2f}%")
        print(f"  Actual accuracy: {actual_accuracy*100:.2f}%")
        print(f"  Calibration error: {calibration_error*100:.2f}% {'(overconfident)' if mean_belief > actual_accuracy else '(underconfident)'}")

    # Detailed calibration by confidence bins
    print(f"\n{'-'*100}")
    print(f"CALIBRATION BY CONFIDENCE LEVEL:")
    print(f"{'-'*100}")
    print(f"{'Confidence Range':<20} {'Count':>8} {'Accuracy':>12} {'Mean Confidence':>18} {'Calibration Gap':>18}")
    print(f"{'-'*100}")

    calibration_data = calculate_calibration(results)
    for bin_label in sorted(calibration_data.keys()):
        data = calibration_data[bin_label]
        accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0
        mean_conf = statistics.mean(data['beliefs']) if data['beliefs'] else 0
        gap = mean_conf - accuracy

        print(f"{bin_label:<20} {data['total']:>8} {accuracy*100:>11.2f}% {mean_conf:>17.3f} {gap:>+17.3f}")


def generate_comparison_table(all_data):
    """Generate comparison table across all models"""
    print(f"\n\n{'='*120}")
    print("MODEL COMPARISON - OVERALL ACCURACY")
    print(f"{'='*120}")

    print(f"\n{'Model':<30} {'Total':>8} {'Correct':>10} {'Accuracy':>12} {'Mean Conf':>12} {'Calib Err':>12}")
    print(f"{'-'*120}")

    # Sort by accuracy
    sorted_models = sorted(all_data.items(), key=lambda x: x[1]['metrics']['accuracy'], reverse=True)

    for model_key, data in sorted_models:
        model_name = next(m['name'] for m in MODELS if m['key'] == model_key)
        metrics = data['metrics']
        belief_metrics = metrics.get('belief', {})

        mean_conf = belief_metrics.get('mean_belief', 0)
        calib_err = abs(mean_conf - metrics['accuracy'])

        print(f"{model_name:<30} {metrics['total']:>8} {metrics['correct']:>10} "
              f"{metrics['accuracy']*100:>11.2f}% {mean_conf:>11.3f} {calib_err*100:>11.2f}%")

    # Comparison by USMLE step
    print(f"\n\n{'='*120}")
    print("MODEL COMPARISON - ACCURACY BY USMLE STEP")
    print(f"{'='*120}")

    for step_name in ['step1', 'step2', 'step3']:
        print(f"\n{step_name.upper()} Questions:")
        print(f"{'-'*120}")
        print(f"{'Model':<30} {'Total':>8} {'Correct':>10} {'Accuracy':>12}")
        print(f"{'-'*120}")

        # Sort by step accuracy
        step_sorted = []
        for model_key, data in all_data.items():
            metrics = data['metrics']
            if step_name in metrics:
                step_sorted.append((model_key, metrics[step_name]))

        step_sorted.sort(key=lambda x: x[1]['accuracy'], reverse=True)

        for model_key, step_data in step_sorted:
            model_name = next(m['name'] for m in MODELS if m['key'] == model_key)
            print(f"{model_name:<30} {step_data['total']:>8} {step_data['correct']:>10} "
                  f"{step_data['accuracy']*100:>11.2f}%")

    # Confidence analysis comparison
    print(f"\n\n{'='*120}")
    print("MODEL COMPARISON - CONFIDENCE AND CALIBRATION")
    print(f"{'='*120}")
    print(f"\n{'Model':<30} {'Conf(All)':>12} {'Conf(✓)':>12} {'Conf(✗)':>12} {'Gap':>12} {'Overconf?':>12}")
    print(f"{'-'*120}")

    for model_key, data in sorted_models:
        model_name = next(m['name'] for m in MODELS if m['key'] == model_key)
        belief_metrics = data['metrics'].get('belief', {})

        mean_all = belief_metrics.get('mean_belief', 0)
        mean_correct = belief_metrics.get('mean_belief_correct', 0)
        mean_incorrect = belief_metrics.get('mean_belief_incorrect', 0)
        gap = mean_correct - mean_incorrect

        accuracy = data['metrics']['accuracy']
        overconf = mean_all - accuracy
        overconf_str = f"{overconf*100:+.2f}%" if overconf != 0 else "calibrated"

        print(f"{model_name:<30} {mean_all:>11.3f} {mean_correct:>11.3f} {mean_incorrect:>11.3f} "
              f"{gap:>+11.3f} {overconf_str:>12}")

    print(f"\n{'='*120}")
    print("LEGEND:")
    print(f"{'='*120}")
    print("Conf(All):    Mean confidence across all answers")
    print("Conf(✓):      Mean confidence when answer was correct")
    print("Conf(✗):      Mean confidence when answer was incorrect")
    print("Gap:          Difference in confidence between correct and incorrect (higher is better)")
    print("Overconf?:    Calibration error (positive = overconfident, negative = underconfident)")
    print("Calib Err:    Absolute calibration error")


def analyze_question_difficulty(all_data):
    """Analyze which questions are hardest/easiest across models"""
    print(f"\n\n{'='*120}")
    print("QUESTION DIFFICULTY ANALYSIS")
    print(f"{'='*120}")

    # Collect data for each question across all models
    question_performance = defaultdict(lambda: {'correct': 0, 'total': 0, 'models_correct': []})

    for model_key, data in all_data.items():
        model_name = next(m['name'] for m in MODELS if m['key'] == model_key)
        for result in data['results']:
            q_id = result['id']
            question_performance[q_id]['total'] += 1
            if result.get('correct', False):
                question_performance[q_id]['correct'] += 1
                question_performance[q_id]['models_correct'].append(model_name)

    # Calculate difficulty
    for q_id, data in question_performance.items():
        data['accuracy'] = data['correct'] / data['total'] if data['total'] > 0 else 0

    # Find hardest questions (answered correctly by fewest models)
    sorted_by_difficulty = sorted(question_performance.items(), key=lambda x: x[1]['accuracy'])

    print(f"\nHARDEST QUESTIONS (answered correctly by fewest models):")
    print(f"{'-'*120}")
    print(f"{'Question ID':<20} {'Models Correct':>15} {'Accuracy':>12} {'Models that got it right':<50}")
    print(f"{'-'*120}")

    for q_id, data in sorted_by_difficulty[:20]:
        models_str = ', '.join(data['models_correct'][:3])
        if len(data['models_correct']) > 3:
            models_str += f" +{len(data['models_correct'])-3} more"
        print(f"{q_id:<20} {data['correct']}/{data['total']:>2} {data['accuracy']*100:>11.1f}% {models_str:<50}")

    print(f"\nEASIEST QUESTIONS (answered correctly by most models):")
    print(f"{'-'*120}")
    print(f"{'Question ID':<20} {'Models Correct':>15} {'Accuracy':>12} {'Models that got it wrong':<50}")
    print(f"{'-'*120}")

    for q_id, data in sorted_by_difficulty[-20:]:
        all_models = set(m['name'] for m in MODELS)
        models_wrong = all_models - set(data['models_correct'])
        models_str = ', '.join(list(models_wrong)[:3]) if models_wrong else "None"
        if len(models_wrong) > 3:
            models_str += f" +{len(models_wrong)-3} more"
        print(f"{q_id:<20} {data['correct']}/{data['total']:>2} {data['accuracy']*100:>11.1f}% {models_str:<50}")


def main():
    """Main analysis function"""
    print("\n" + "="*120)
    print("MODEL ACCURACY ANALYSIS - USMLE SAMPLE DATASET")
    print("="*120)
    print("\nAnalyzing test results for all models...")

    all_data = {}

    # Load all test results
    for model_info in MODELS:
        filepath = os.path.join(TESTS_DIR, model_info['file'])
        data = load_test_results(filepath)

        if data:
            all_data[model_info['key']] = data
            print(f"  ✓ Loaded {model_info['name']}: {data['metrics']['total']} questions")
        else:
            print(f"  ✗ Failed to load {model_info['name']}")

    if not all_data:
        print("\nNo test results found!")
        return

    # Detailed analysis for each model
    for model_info in MODELS:
        if model_info['key'] in all_data:
            print_model_analysis(model_info, all_data[model_info['key']])

    # Generate comparison tables
    generate_comparison_table(all_data)

    # Question difficulty analysis
    analyze_question_difficulty(all_data)

    print(f"\n{'='*120}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*120}\n")


if __name__ == '__main__':
    main()
