import argparse
import json
import re
from collections import defaultdict

def extract_answer(text):
    """Extract the answer letter from agent's response.

    Looks for pattern: "ANSWER: [Letter]" or "ANSWER: [Letter]."
    """
    if not text:
        return None

    # Try to find "ANSWER: X" pattern (case insensitive)
    patterns = [
        r'ANSWER:\s*([A-E])',  # ANSWER: A
        r'ANSWER:\s*([A-E])\.',  # ANSWER: A.
        r'ANSWER:\s*([A-E])\n',  # ANSWER: A\n
        r'answer:\s*([A-E])',  # answer: A (lowercase)
        r'\*\*ANSWER:\s*([A-E])',  # **ANSWER: A
        r'Final Answer:\s*([A-E])',  # Final Answer: A
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate accuracy of agent recommendations on USMLE data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        default=['output/usmle/principal_deepseek_all.json', 'output/usmle/principal_deepseek_usmle.json', 'output/usmle/principal_llama-large_bayesian.json'],
        help='Path(s) to input USMLE principal JSON file(s)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis/usmle_accuracy',
        help='Path prefix for output files'
    )
    args = parser.parse_args()

    # Group input files by agent model name
    from collections import defaultdict
    files_by_model = defaultdict(list)

    for input_file in args.input:
        # Extract agent model name from filename
        filename = input_file.split('/')[-1].replace('principal_', '').replace('.json', '')
        # Remove suffixes like _all, _usmle, _bayesian to get base model name
        for suffix in ['_all', '_usmle', '_bayesian', '_train', '_test']:
            if filename.endswith(suffix):
                filename = filename[:-len(suffix)]
                break
        files_by_model[filename].append(input_file)

    # Process all model groups
    all_results = {}

    for model_name, input_files in files_by_model.items():
        print(f"\nEvaluating model: {model_name}")
        print(f"  Input files: {input_files}")

        # Merge all files for this model
        all_data = []
        for input_file in input_files:
            with open(input_file, 'r') as f:
                raw_data = json.load(f)
                all_data.extend(raw_data)

        # Group by principal type
        principal_data = defaultdict(list)
        for item in all_data:
            principal_name = item['principal_name']
            principal_data[principal_name].append(item)

        # Find common cases across all principals (if bayesian exists)
        if 'bayesian_principal' in principal_data:
            # Get cases that exist in ALL principals
            all_principals_list = list(principal_data.keys())
            common_case_ids = set(item['case_id'] for item in principal_data[all_principals_list[0]])
            for principal_name in all_principals_list[1:]:
                case_ids = set(item['case_id'] for item in principal_data[principal_name])
                common_case_ids &= case_ids

            print(f"  Total cases: {len(set(item['case_id'] for item in all_data))}")
            print(f"  Common cases across all principals: {len(common_case_ids)}")
        else:
            # If no bayesian, just use all cases
            common_case_ids = set(item['case_id'] for item in all_data)
            print(f"  Total cases: {len(common_case_ids)}")

        # Calculate accuracy for each principal (only on common cases)
        model_results = {}

        for principal_name, items in principal_data.items():
            # Filter to only common cases
            filtered_items = [item for item in items if item['case_id'] in common_case_ids]

            total = len(filtered_items)
            correct = 0
            incorrect = 0
            unparseable = 0
            case_details = {}

            for item in filtered_items:
                case_id = item['case_id']
                agent_response = item.get('information', '')
                correct_answer = item.get('correct_answer_idx', '').upper()

                # Extract agent's answer
                agent_answer = extract_answer(agent_response)

                # Compare
                is_correct = None
                if agent_answer is None:
                    unparseable += 1
                    is_correct = None
                elif agent_answer == correct_answer:
                    correct += 1
                    is_correct = True
                else:
                    incorrect += 1
                    is_correct = False

                case_details[case_id] = {
                    'agent_answer': agent_answer,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct
                }

            accuracy = correct / total if total > 0 else 0

            model_results[principal_name] = {
                'total_cases': total,
                'correct': correct,
                'incorrect': incorrect,
                'unparseable': unparseable,
                'accuracy': accuracy,
                'case_details': case_details
            }

            print(f"  {principal_name}: {accuracy:.2%} accuracy ({correct}/{total} correct, {unparseable} unparseable)")

        all_results[model_name] = model_results

        # Cross-analyze with principal decisions (accept/reject)
        # This shows how principals respond to correct vs incorrect answers
        print(f"\n  Analyzing accept/reject patterns for {model_name}:")

        for principal_name, items in principal_data.items():
            filtered_items = [item for item in items if item['case_id'] in common_case_ids]

            # Track patterns
            correct_accepted = 0
            correct_rejected = 0
            incorrect_accepted = 0
            incorrect_rejected = 0
            unparseable_accepted = 0
            unparseable_rejected = 0

            for item in filtered_items:
                case_id = item['case_id']
                agent_response = item.get('information', '')
                correct_answer = item.get('correct_answer_idx', '').upper()
                principal_decision = item.get('decision', '').strip().lower()

                # Extract agent's answer
                agent_answer = extract_answer(agent_response)

                # Determine if accepted
                is_accepted = 'accept' in principal_decision

                # Categorize
                if agent_answer is None:
                    if is_accepted:
                        unparseable_accepted += 1
                    else:
                        unparseable_rejected += 1
                elif agent_answer == correct_answer:
                    if is_accepted:
                        correct_accepted += 1
                    else:
                        correct_rejected += 1
                else:
                    if is_accepted:
                        incorrect_accepted += 1
                    else:
                        incorrect_rejected += 1

            # Add to model results
            model_results[principal_name]['acceptance_patterns'] = {
                'correct_accepted': correct_accepted,
                'correct_rejected': correct_rejected,
                'incorrect_accepted': incorrect_accepted,
                'incorrect_rejected': incorrect_rejected,
                'unparseable_accepted': unparseable_accepted,
                'unparseable_rejected': unparseable_rejected
            }

            total_correct = correct_accepted + correct_rejected
            total_incorrect = incorrect_accepted + incorrect_rejected

            if total_correct > 0:
                correct_accept_rate = correct_accepted / total_correct
            else:
                correct_accept_rate = 0

            if total_incorrect > 0:
                incorrect_accept_rate = incorrect_accepted / total_incorrect
            else:
                incorrect_accept_rate = 0

            print(f"    {principal_name}:")
            print(f"      Correct answers:   {correct_accepted} accepted, {correct_rejected} rejected (accept rate: {correct_accept_rate:.1%})")
            print(f"      Incorrect answers: {incorrect_accepted} accepted, {incorrect_rejected} rejected (accept rate: {incorrect_accept_rate:.1%})")

        all_results[model_name] = model_results

    # Save results to JSON
    output_data = {
        'results': all_results
    }

    with open(f'{args.output}.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved accuracy results: {args.output}.json")

    # Create summary comparison if bayesian exists
    print("\n" + "="*80)
    print("SUMMARY: Accuracy Comparison")
    print("="*80)

    for model_name, model_results in all_results.items():
        if 'bayesian_principal' in model_results:
            bayesian_acc = model_results['bayesian_principal']['accuracy']
            print(f"\n{model_name.upper()}:")
            print(f"  Bayesian (baseline): {bayesian_acc:.2%}")

            # Show differences from bayesian
            for principal_name, results in sorted(model_results.items()):
                if principal_name != 'bayesian_principal':
                    acc = results['accuracy']
                    diff = acc - bayesian_acc
                    sign = "+" if diff > 0 else ""
                    print(f"  {principal_name.replace('_principal', ''):20s}: {acc:.2%} ({sign}{diff:.2%})")
        else:
            print(f"\n{model_name.upper()}:")
            for principal_name, results in model_results.items():
                acc = results['accuracy']
                print(f"  {principal_name}: {acc:.2%}")

    # Create summary of acceptance patterns
    print("\n" + "="*80)
    print("SUMMARY: Acceptance Patterns (Correct vs Incorrect Answers)")
    print("="*80)

    for model_name, model_results in all_results.items():
        if 'bayesian_principal' in model_results:
            print(f"\n{model_name.upper()}:")
            print(f"  {'Principal':<20s} {'Correct Accept':<20s} {'Incorrect Accept':<20s} {'Error Rate':<15s}")
            print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*15}")

            for principal_name in sorted(model_results.keys()):
                results = model_results[principal_name]
                patterns = results.get('acceptance_patterns', {})

                correct_accepted = patterns.get('correct_accepted', 0)
                correct_rejected = patterns.get('correct_rejected', 0)
                incorrect_accepted = patterns.get('incorrect_accepted', 0)
                incorrect_rejected = patterns.get('incorrect_rejected', 0)

                total_correct = correct_accepted + correct_rejected
                total_incorrect = incorrect_accepted + incorrect_rejected

                if total_correct > 0:
                    correct_accept_rate = correct_accepted / total_correct
                else:
                    correct_accept_rate = 0

                if total_incorrect > 0:
                    incorrect_accept_rate = incorrect_accepted / total_incorrect
                else:
                    incorrect_accept_rate = 0

                # Error rate: accepting incorrect + rejecting correct
                total_cases = total_correct + total_incorrect
                errors = incorrect_accepted + correct_rejected
                error_rate = errors / total_cases if total_cases > 0 else 0

                name_display = principal_name.replace('_principal', '')
                correct_str = f"{correct_accepted}/{total_correct} ({correct_accept_rate:.1%})"
                incorrect_str = f"{incorrect_accepted}/{total_incorrect} ({incorrect_accept_rate:.1%})"
                error_str = f"{errors}/{total_cases} ({error_rate:.1%})"

                print(f"  {name_display:<20s} {correct_str:<20s} {incorrect_str:<20s} {error_str:<15s}")

            print(f"\n  Key:")
            print(f"    - Correct Accept: # of correct answers accepted / total correct")
            print(f"    - Incorrect Accept: # of incorrect answers accepted / total incorrect")
            print(f"    - Error Rate: (incorrect accepted + correct rejected) / total cases")
        else:
            print(f"\n{model_name.upper()}:")
            for principal_name, results in model_results.items():
                patterns = results.get('acceptance_patterns', {})
                print(f"  {principal_name}:")
                print(f"    Correct answers: {patterns.get('correct_accepted', 0)} accepted, {patterns.get('correct_rejected', 0)} rejected")
                print(f"    Incorrect answers: {patterns.get('incorrect_accepted', 0)} accepted, {patterns.get('incorrect_rejected', 0)} rejected")
