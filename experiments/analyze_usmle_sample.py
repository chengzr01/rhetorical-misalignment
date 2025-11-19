import argparse
import json
import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze decision-making from USMLE sample data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='output/usmle_sample',
        help='Directory containing USMLE sample principal JSON files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis/usmle_sample_decision_making_analysis',
        help='Path prefix for output files'
    )
    parser.add_argument(
        '--case-dir',
        type=str,
        default='cases/usmle_sample',
        help='Directory for saving max difference cases'
    )
    parser.add_argument(
        '--belief-file',
        type=str,
        default='tests/test_usmle_sample_deepseek-ai-deepseek-v3.1_belief.json',
        help='Path to belief file from DeepSeek-v3.1 model'
    )
    parser.add_argument(
        '--belief-threshold',
        type=float,
        default=0.8,
        help='Only include cases where DeepSeek-v3.1 belief is below this threshold'
    )
    args = parser.parse_args()

    # Load belief data if available
    belief_data = {}
    if args.belief_file and os.path.exists(args.belief_file):
        print(f"Loading belief data from: {args.belief_file}")
        with open(args.belief_file, 'r') as f:
            belief_json = json.load(f)
            for result in belief_json.get('results', []):
                case_id = result.get('id')
                belief = result.get('belief')
                predicted_answer = result.get('predicted_answer_idx')
                correct = result.get('correct', False)
                if case_id and belief is not None:
                    belief_data[case_id] = {
                        'belief': belief,
                        'predicted_answer': predicted_answer,
                        'correct': correct
                    }
        print(f"  Loaded beliefs for {len(belief_data)} cases")
        print(f"  Belief threshold: {args.belief_threshold}")
    else:
        print(f"Belief file not found or not specified: {args.belief_file}")

    # Automatically discover all models with bayesian and behavioral files
    files_by_model = {}

    # Find all bayesian and behavioral files
    bayesian_files = glob.glob(os.path.join(args.input_dir, 'principal_*_bayesian.json'))
    behavioral_files = glob.glob(os.path.join(args.input_dir, 'principal_*_behavioral.json'))

    # Group files by model name
    for filepath in bayesian_files + behavioral_files:
        filename = os.path.basename(filepath)
        # Extract model name: principal_MODEL_bayesian.json -> MODEL
        if '_bayesian.json' in filename:
            model_name = filename.replace('principal_', '').replace('_bayesian.json', '')
        elif '_behavioral.json' in filename:
            model_name = filename.replace('principal_', '').replace('_behavioral.json', '')
        else:
            continue

        if model_name not in files_by_model:
            files_by_model[model_name] = []
        files_by_model[model_name].append(filepath)

    # Process all model groups
    all_results = {}

    for model_name, input_files in files_by_model.items():
        print(f"\nProcessing model: {model_name}")
        print(f"  Input files: {input_files}")

        # Merge all files for this model
        data = {}
        for input_file in input_files:
            with open(input_file, 'r') as f:
                raw_data = json.load(f)

            # Convert USMLE format (flat list) to nested format
            # USMLE format: [{principal_name: ..., case_id: ..., decision: ..., ...}, ...]
            # Target format: {case_id: {principal_type: {analysis: [{decision: ...}]}}}
            for item in raw_data:
                case_id = item['case_id']
                principal_name = item['principal_name']
                decision = item.get('decision', '')
                reasoning = item.get('reasoning', '')

                if case_id not in data:
                    data[case_id] = {}

                if principal_name not in data[case_id]:
                    data[case_id][principal_name] = {
                        'analysis': []
                    }

                # Each item in USMLE is a single decision
                data[case_id][principal_name]['analysis'].append({
                    'decision': decision,
                    'reasoning': reasoning
                })

        # First pass: collect all case data
        # Only include bayesian and behavioral principals
        allowed_principals = {
            'bayesian_principal',
            'behavioral_principal'
        }

        all_principal_data = {}
        for case_id, case_data in data.items():
            for principal_type, principal_response in case_data.items():
                # Filter to only allowed principals
                if principal_type not in allowed_principals:
                    continue

                if principal_type not in all_principal_data:
                    all_principal_data[principal_type] = {}

                # Get decision for this case
                analysis_blocks = principal_response.get("analysis", [])
                for block in analysis_blocks:
                    decision = block.get("decision", "").strip().lower()
                    is_accept = "accept" in decision
                    all_principal_data[principal_type][case_id] = {
                        'decision': 'accept' if is_accept else 'reject',
                        'reasoning': block.get('reasoning', '')
                    }

        # Find common cases across all principals (only if bayesian exists)
        if 'bayesian_principal' in all_principal_data:
            # Get cases that exist in ALL principals
            all_principals_list = list(all_principal_data.keys())
            common_cases = set(all_principal_data[all_principals_list[0]].keys())
            for principal_type in all_principals_list[1:]:
                common_cases &= set(all_principal_data[principal_type].keys())

            print(f"  Total cases found: {len(data)}")
            print(f"  Common cases across all principals: {len(common_cases)}")
        else:
            # If no bayesian, just use all cases
            common_cases = set(data.keys())
            print(f"  Total cases found: {len(data)}")

        # Count acceptance rates by principal type (only for common cases)
        principal_stats = {}
        for principal_type, case_data in all_principal_data.items():
            principal_stats[principal_type] = {
                'total_cases': 0,
                'accept_cases': 0,
                'reject_cases': 0,
                'case_details': {}
            }

            for case_id in common_cases:
                if case_id in case_data:
                    decision_data = case_data[case_id]
                    principal_stats[principal_type]['case_details'][case_id] = decision_data
                    principal_stats[principal_type]['total_cases'] += 1

                    if decision_data['decision'] == 'accept':
                        principal_stats[principal_type]['accept_cases'] += 1
                    else:
                        principal_stats[principal_type]['reject_cases'] += 1

        # Calculate acceptance rates and store results
        model_results = {}
        for principal_type, principal_stats_item in principal_stats.items():
            total_cases = principal_stats_item['total_cases']
            if total_cases > 0:
                acceptance_rate = principal_stats_item['accept_cases'] / total_cases
                model_results[principal_type] = {
                    'total_cases': total_cases,
                    'accept_cases': principal_stats_item['accept_cases'],
                    'reject_cases': principal_stats_item['reject_cases'],
                    'acceptance_rate': acceptance_rate,
                    'case_details': principal_stats_item['case_details']
                }
                print(f"{model_name} - {principal_type}: {acceptance_rate:.2%} acceptance rate ({principal_stats_item['accept_cases']}/{total_cases} cases)")

        all_results[model_name] = model_results

        # Find cases where bayesian_principal and behavioral_principal differ
        if 'bayesian_principal' in model_results and 'behavioral_principal' in model_results:
            bayesian_case_details = model_results['bayesian_principal']['case_details']
            behavioral_case_details = model_results['behavioral_principal']['case_details']

            # Find cases where the two principals differ in decision
            case_differences = []

            for case_id in bayesian_case_details:
                if case_id in behavioral_case_details:
                    bayesian_decision = bayesian_case_details[case_id]['decision']
                    behavioral_decision = behavioral_case_details[case_id]['decision']

                    # Only include cases where they differ
                    if bayesian_decision != behavioral_decision:
                        case_info = {
                            'case_id': case_id,
                            'bayesian_decision': bayesian_decision,
                            'behavioral_decision': behavioral_decision,
                            'bayesian_reasoning': bayesian_case_details[case_id]['reasoning'],
                            'behavioral_reasoning': behavioral_case_details[case_id]['reasoning']
                        }

                        # Add belief data if available
                        if case_id in belief_data:
                            case_info['deepseek_belief'] = belief_data[case_id]['belief']
                            case_info['deepseek_predicted_answer'] = belief_data[case_id]['predicted_answer']
                            case_info['deepseek_correct'] = belief_data[case_id]['correct']

                            # Only include cases with low belief if belief data is available
                            if belief_data[case_id]['belief'] < args.belief_threshold:
                                case_differences.append(case_info)
                        else:
                            # If no belief data, include all differing cases
                            case_differences.append(case_info)

            # Sort by case_id for consistency
            case_differences.sort(key=lambda x: x['case_id'])

            # Convert model_name: replace hyphens with underscores for filename
            model_name_for_file = model_name.replace('-', '_')

            # Save filtered cases (only low belief if belief data available)
            model_max_diff_data = {
                'model': model_name,
                'belief_threshold': args.belief_threshold if belief_data else None,
                'total_differing_cases': len(case_differences),
                'cases': case_differences
            }

            os.makedirs(args.case_dir, exist_ok=True)
            case_output_file = f'{args.case_dir}/decision_making_analysis_{model_name_for_file}_max_diff.json'
            with open(case_output_file, 'w') as f:
                json.dump(model_max_diff_data, f, indent=2)

            if belief_data:
                print(f"Saved max difference cases: {case_output_file} ({len(case_differences)} cases with belief < {args.belief_threshold})")
            else:
                print(f"Saved max difference cases: {case_output_file} ({len(case_differences)} differing cases)")

    # Save all results to JSON
    output_data = {
        'results': all_results
    }

    with open(f'{args.output}.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved results: {args.output}.json")
