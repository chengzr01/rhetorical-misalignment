import argparse
import json
import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze decision-making from USMLE data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='output/usmle',
        help='Directory containing USMLE principal JSON files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis/usmle_decision_making_analysis',
        help='Path prefix for output files'
    )
    parser.add_argument(
        '--case-dir',
        type=str,
        default='cases/usmle',
        help='Directory for saving max difference cases'
    )
    args = parser.parse_args()

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
                        case_differences.append({
                            'case_id': case_id,
                            'bayesian_decision': bayesian_decision,
                            'behavioral_decision': behavioral_decision,
                            'bayesian_reasoning': bayesian_case_details[case_id]['reasoning'],
                            'behavioral_reasoning': behavioral_case_details[case_id]['reasoning']
                        })

            # Sort by case_id for consistency
            case_differences.sort(key=lambda x: x['case_id'])

            model_max_diff_data = {
                'model': model_name,
                'total_differing_cases': len(case_differences),
                'cases': case_differences
            }

            # Save to cases directory
            os.makedirs(args.case_dir, exist_ok=True)
            case_output_file = f'{args.case_dir}/decision_making_analysis_{model_name}_max_diff.json'
            with open(case_output_file, 'w') as f:
                json.dump(model_max_diff_data, f, indent=2)
            print(f"Saved max difference cases: {case_output_file} ({len(case_differences)} differing cases)")

    # Save all results to JSON
    output_data = {
        'results': all_results
    }

    with open(f'{args.output}.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved results: {args.output}.json")