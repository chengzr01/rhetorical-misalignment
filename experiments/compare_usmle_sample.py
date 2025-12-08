import argparse
import json
import glob
import os
import re
import statistics

def extract_model_name_from_test_file(filename):
    """Extract model name from test filename.

    Expected format: test_usmle_sample_{model_name}_belief.json
    Example: test_usmle_sample_deepseek-deepseek-chat-v3.1_belief.json -> deepseek-deepseek-chat-v3.1
    """
    basename = os.path.basename(filename)
    # Remove prefix and suffix
    match = re.match(r'test_usmle_sample_(.+)_belief\.json', basename)
    if match:
        return match.group(1)
    return None

def load_test_results(test_dir):
    """Load all test result files and organize by model name and case_id."""
    test_files = glob.glob(os.path.join(test_dir, 'test_usmle_sample_*_belief.json'))

    model_results = {}

    for test_file in test_files:
        model_name = extract_model_name_from_test_file(test_file)
        if not model_name:
            print(f"Warning: Could not extract model name from {test_file}")
            continue

        print(f"Loading test results for model: {model_name}")
        with open(test_file, 'r') as f:
            test_data = json.load(f)

        # Organize results by case_id for quick lookup
        case_results = {}
        for result in test_data.get('results', []):
            case_id = result.get('id')
            if case_id:
                case_results[case_id] = {
                    'correct': result.get('correct', False),
                    'predicted_answer_idx': result.get('predicted_answer_idx'),
                    'correct_answer_idx': result.get('correct_answer_idx'),
                    'belief': result.get('belief'),
                    'meta_info': result.get('meta_info')
                }

        model_results[model_name] = case_results
        print(f"  Loaded {len(case_results)} test results for {model_name}")

    return model_results

def calculate_model_accuracies(cases, model_results):
    """Calculate accuracy and belief statistics for each model on the given cases."""
    model_accuracies = {}

    # Extract case IDs from the manipulative cases
    case_ids = [case['case_id'] for case in cases]

    print(f"\nCalculating accuracies for {len(case_ids)} manipulative cases")

    for model_name, case_results in model_results.items():
        total = 0
        correct = 0
        missing = 0

        case_details = {}
        all_beliefs = []
        correct_beliefs = []
        incorrect_beliefs = []

        for case_id in case_ids:
            if case_id in case_results:
                result = case_results[case_id]
                total += 1
                belief = result['belief']

                if result['correct']:
                    correct += 1
                    if belief is not None:
                        correct_beliefs.append(belief)
                else:
                    if belief is not None:
                        incorrect_beliefs.append(belief)

                if belief is not None:
                    all_beliefs.append(belief)

                case_details[case_id] = {
                    'correct': result['correct'],
                    'predicted_answer': result['predicted_answer_idx'],
                    'correct_answer': result['correct_answer_idx'],
                    'belief': belief
                }
            else:
                missing += 1

        accuracy = correct / total if total > 0 else 0.0

        # Calculate belief statistics
        belief_stats = {}
        if all_beliefs:
            belief_stats['mean'] = statistics.mean(all_beliefs)
            belief_stats['median'] = statistics.median(all_beliefs)
            belief_stats['stdev'] = statistics.stdev(all_beliefs) if len(all_beliefs) > 1 else 0.0
            belief_stats['min'] = min(all_beliefs)
            belief_stats['max'] = max(all_beliefs)

        if correct_beliefs:
            belief_stats['mean_correct'] = statistics.mean(correct_beliefs)
            belief_stats['median_correct'] = statistics.median(correct_beliefs)
            belief_stats['stdev_correct'] = statistics.stdev(correct_beliefs) if len(correct_beliefs) > 1 else 0.0

        if incorrect_beliefs:
            belief_stats['mean_incorrect'] = statistics.mean(incorrect_beliefs)
            belief_stats['median_incorrect'] = statistics.median(incorrect_beliefs)
            belief_stats['stdev_incorrect'] = statistics.stdev(incorrect_beliefs) if len(incorrect_beliefs) > 1 else 0.0

        model_accuracies[model_name] = {
            'total_cases': total,
            'correct_cases': correct,
            'incorrect_cases': total - correct,
            'missing_cases': missing,
            'accuracy': accuracy,
            'belief_stats': belief_stats,
            'case_details': case_details
        }

        print(f"  {model_name}: {accuracy:.2%} ({correct}/{total} correct, {missing} missing)")

    return model_accuracies

def add_accuracy_to_cases(cases, model_results):
    """Add accuracy information to each case."""
    enhanced_cases = []

    for case in cases:
        case_id = case['case_id']
        enhanced_case = case.copy()

        # Add model-specific accuracy information
        model_predictions = {}
        meta_info = None
        for model_name, case_results in model_results.items():
            if case_id in case_results:
                result = case_results[case_id]
                model_predictions[model_name] = {
                    'correct': result['correct'],
                    'predicted_answer': result['predicted_answer_idx'],
                    'correct_answer': result['correct_answer_idx'],
                    'belief': result['belief']
                }
                # Get meta_info (should be same across all models for a given case)
                if meta_info is None and result.get('meta_info'):
                    meta_info = result['meta_info']

        enhanced_case['model_predictions'] = model_predictions
        if meta_info:
            enhanced_case['meta_info'] = meta_info
        enhanced_cases.append(enhanced_case)

    return enhanced_cases

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare model accuracies on manipulative USMLE cases.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--case-file',
        type=str,
        required=True,
        help='Path to the manipulative cases JSON file'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default='tests',
        help='Directory containing test result JSON files'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path for output file (default: overwrites input file)'
    )
    args = parser.parse_args()

    # Load the case file
    print(f"Loading manipulative cases from: {args.case_file}")
    with open(args.case_file, 'r') as f:
        case_data = json.load(f)

    cases = case_data.get('cases', [])
    print(f"Found {len(cases)} manipulative cases")

    # Load test results
    model_results = load_test_results(args.test_dir)

    if not model_results:
        print("Error: No test results found")
        exit(1)

    # Calculate accuracies for each model
    model_accuracies = calculate_model_accuracies(cases, model_results)

    # Add accuracy information to each case
    enhanced_cases = add_accuracy_to_cases(cases, model_results)

    # Create output data
    output_data = case_data.copy()
    output_data['cases'] = enhanced_cases
    output_data['model_accuracies'] = model_accuracies

    # Determine output path
    output_path = args.output if args.output else args.case_file

    # Save to output file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved enhanced cases with accuracy information to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Model Accuracies and Belief Statistics on Manipulative Cases")
    print("="*80)
    for model_name, stats in sorted(model_accuracies.items()):
        print(f"\n{model_name}:")
        print(f"  Accuracy: {stats['accuracy']:.2%}")
        print(f"  Correct: {stats['correct_cases']}/{stats['total_cases']}")
        print(f"  Incorrect: {stats['incorrect_cases']}/{stats['total_cases']}")
        if stats['missing_cases'] > 0:
            print(f"  Missing: {stats['missing_cases']} cases not found in test results")

        # Print belief statistics
        belief_stats = stats.get('belief_stats', {})
        if belief_stats:
            print(f"\n  Belief Statistics:")
            print(f"    Overall:")
            print(f"      Mean: {belief_stats.get('mean', 0):.4f}")
            print(f"      Median: {belief_stats.get('median', 0):.4f}")
            print(f"      Std Dev: {belief_stats.get('stdev', 0):.4f}")
            print(f"      Range: [{belief_stats.get('min', 0):.4f}, {belief_stats.get('max', 0):.4f}]")

            if 'mean_correct' in belief_stats:
                print(f"    Correct Predictions:")
                print(f"      Mean: {belief_stats.get('mean_correct', 0):.4f}")
                print(f"      Median: {belief_stats.get('median_correct', 0):.4f}")
                print(f"      Std Dev: {belief_stats.get('stdev_correct', 0):.4f}")

            if 'mean_incorrect' in belief_stats:
                print(f"    Incorrect Predictions:")
                print(f"      Mean: {belief_stats.get('mean_incorrect', 0):.4f}")
                print(f"      Median: {belief_stats.get('median_incorrect', 0):.4f}")
                print(f"      Std Dev: {belief_stats.get('stdev_incorrect', 0):.4f}")

            # Calculate calibration gap
            if 'mean_correct' in belief_stats and 'mean_incorrect' in belief_stats:
                calibration_gap = belief_stats['mean_correct'] - belief_stats['mean_incorrect']
                print(f"    Calibration Gap (correct - incorrect): {calibration_gap:.4f}")
