import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze decision-making from USMLE data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        default=['output/usmle/principal_deepseek_all.json', 'output/usmle/principal_deepseek_usmle.json', 'output/usmle/principal_llama-large_bayesian.json'],
        help='Path(s) to input USMLE principal JSON file(s). Files will be processed separately unless merged.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis/usmle_decision_making_analysis',
        help='Path prefix for output files'
    )
    args = parser.parse_args()

    # Group input files by agent model name
    # e.g., principal_deepseek_all.json and principal_deepseek_usmle.json -> deepseek
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
        all_principal_data = {}
        for case_id, case_data in data.items():
            for principal_type, principal_response in case_data.items():
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
        
        # Find cases with maximum difference between non-Bayesian and Bayesian principals
        if 'bayesian_principal' in model_results:
            bayesian_case_details = model_results['bayesian_principal']['case_details']

            # Collect all principal differences in one structure
            model_max_diff_data = {
                'model': model_name,
                'principals': {}
            }

            # For each non-Bayesian principal, find cases with different decisions
            for principal_type in model_results:
                if principal_type != 'bayesian_principal':
                    principal_case_details = model_results[principal_type]['case_details']

                    # Calculate differences for each case
                    # For USMLE: binary decisions, so difference is 1.0 if different, 0.0 if same
                    case_differences = {}
                    for case_id in principal_case_details:
                        if case_id in bayesian_case_details:
                            principal_decision = principal_case_details[case_id]['decision']
                            bayesian_decision = bayesian_case_details[case_id]['decision']

                            # Convert to binary acceptance rate (1.0 for accept, 0.0 for reject)
                            principal_rate = 1.0 if principal_decision == 'accept' else 0.0
                            bayesian_rate = 1.0 if bayesian_decision == 'accept' else 0.0
                            difference = abs(principal_rate - bayesian_rate)

                            if difference > 0:  # Only include cases where they differ
                                case_differences[case_id] = {
                                    'difference': difference,
                                    'principal_decision': principal_decision,
                                    'bayesian_decision': bayesian_decision,
                                    'principal_rate': principal_rate,
                                    'bayesian_rate': bayesian_rate
                                }

                    # Sort by case_id for consistency (all differences are 1.0)
                    sorted_cases = sorted(case_differences.items(), key=lambda x: x[0])

                    # Store all differing cases for this principal
                    model_max_diff_data['principals'][principal_type] = [
                        {
                            'case_id': case_id,
                            **case_info
                        }
                        for case_id, case_info in sorted_cases
                    ]

            # Save one file per model with all principals' differing cases
            output_file = f'{args.output}_{model_name}_max_diff.json'
            with open(output_file, 'w') as f:
                json.dump(model_max_diff_data, f, indent=2)

            print(f"Saved max difference cases for {model_name}: {output_file}")

            # Also save to cases/usmle folder
            import os
            case_dir = 'cases/usmle'
            os.makedirs(case_dir, exist_ok=True)
            case_output_file = f'{case_dir}/{model_name}_max_diff.json'
            with open(case_output_file, 'w') as f:
                json.dump(model_max_diff_data, f, indent=2)
            print(f"Saved max difference cases to cases folder: {case_output_file}")
    
    # Create plots for each model
    for model_name, model_results in all_results.items():
        # Create single plot showing acceptance rates
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Define the desired order
        desired_order = ["bayesian_principal", "anchoring_principal", "availability_principal", "confirmation_principal", "conservatism_principal", "overconfidence_principal", "prospect_principal"]

        # Filter to only include principal types that exist in the data
        principal_types = [pt for pt in desired_order if pt in model_results]
        acceptance_rates = [model_results[pt]['acceptance_rate'] for pt in principal_types]
        display_labels = [pt.replace('_principal', '') for pt in principal_types]

        # Create color array
        colors = []
        for pt in principal_types:
            if pt == 'bayesian_principal':
                colors.append('blue')
            else:
                colors.append('gray')

        # Plot acceptance rates
        bars = ax.bar(range(len(principal_types)), acceptance_rates, color=colors, alpha=0.7)

        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, acceptance_rates)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1%}',
                   ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Principal Type', fontsize=12)
        ax.set_ylabel('Acceptance Rate', fontsize=12)
        ax.set_title(f'USMLE Acceptance Rates - {model_name}\n(Blue = Bayesian Baseline)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(principal_types)))
        ax.set_xticklabels(display_labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{args.output}_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nSaved plot: {args.output}_{model_name}.png")
    
    # Save all results to JSON
    output_data = {
        'results': all_results
    }

    with open(f'{args.output}.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved results: {args.output}.json")