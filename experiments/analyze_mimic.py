import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze decision-making from MIMIC-IV episodes data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        default=['analysis/oss_analysis.json', 'analysis/llama_analysis.json', 'analysis/deepseek_analysis.json', 'analysis/small_sft_analysis.json', 'analysis/small_dpo_analysis.json'],
        help='Path(s) to input decision-making analysis JSON file(s)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis/decision_making_analysis',
        help='Path prefix for output files'
    )
    parser.add_argument(
        '--coarse-threshold',
        type=float,
        default=0.5,
        help='Threshold for coarse acceptance rate (default: 0.5)'
    )
    args = parser.parse_args()
    
    # Process all input files
    all_results = {}
    
    for input_file in args.input:
        # Extract model name from filename
        model_name = input_file.split('/')[-1].replace('_analysis.json', '')
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Count acceptance rates by principal type
        principal_stats = {}
        
        for case_id, case_data in data.items():
            for principal_type, principal_response in case_data.items():
                if principal_type not in principal_stats:
                    principal_stats[principal_type] = {
                        'total_decisions': 0,
                        'accept_decisions': 0,
                        'reject_decisions': 0,
                        'decisions_per_case': [],
                        'total_cases': 0,
                        'coarse_accept_cases': 0,
                        'case_details': {}  # Store case-level details
                    }
                
                # Analyze each decision block
                analysis_blocks = principal_response.get("analysis", [])
                case_accepts = 0
                case_total = 0
                for block in analysis_blocks:
                    decision = block.get("decision", "").strip().lower()
                    principal_stats[principal_type]['total_decisions'] += 1
                    case_total += 1
                    
                    if "accept" in decision:
                        principal_stats[principal_type]['accept_decisions'] += 1
                        case_accepts += 1
                    elif "reject" in decision:
                        principal_stats[principal_type]['reject_decisions'] += 1
                
                # Store per-case acceptance rate
                if case_total > 0:
                    case_acceptance_rate = case_accepts / case_total
                    principal_stats[principal_type]['decisions_per_case'].append(case_acceptance_rate)
                    principal_stats[principal_type]['total_cases'] += 1
                    
                    # Store case details
                    principal_stats[principal_type]['case_details'][case_id] = {
                        'acceptance_rate': case_acceptance_rate,
                        'accepts': case_accepts,
                        'total': case_total
                    }
                    
                    # Coarse acceptance: count case as accept if acceptance rate >= threshold
                    if case_acceptance_rate >= args.coarse_threshold:
                        principal_stats[principal_type]['coarse_accept_cases'] += 1
        
        # Calculate acceptance rates and store results
        model_results = {}
        for principal_type, principal_stats_item in principal_stats.items():
            total = principal_stats_item['total_decisions']
            total_cases = principal_stats_item['total_cases']
            if total > 0:
                acceptance_rate = principal_stats_item['accept_decisions'] / total
                coarse_acceptance_rate = principal_stats_item['coarse_accept_cases'] / total_cases if total_cases > 0 else 0
                model_results[principal_type] = {
                    'total_decisions': total,
                    'accept_decisions': principal_stats_item['accept_decisions'],
                    'reject_decisions': principal_stats_item['reject_decisions'],
                    'acceptance_rate': acceptance_rate,
                    'decisions_per_case': principal_stats_item['decisions_per_case'],
                    'total_cases': total_cases,
                    'coarse_accept_cases': principal_stats_item['coarse_accept_cases'],
                    'coarse_acceptance_rate': coarse_acceptance_rate,
                    'case_details': principal_stats_item['case_details']
                }
                print(f"{model_name} - {principal_type}: {acceptance_rate:.2%} fine acceptance rate ({principal_stats_item['accept_decisions']}/{total}), {coarse_acceptance_rate:.2%} coarse acceptance rate ({principal_stats_item['coarse_accept_cases']}/{total_cases} cases)")
        
        all_results[model_name] = model_results
        
        # Find cases with maximum difference between non-Bayesian and Bayesian principals
        if 'bayesian' in model_results:
            bayesian_case_details = model_results['bayesian']['case_details']
            
            # Collect all principal differences in one structure
            model_max_diff_data = {
                'model': model_name,
                'principals': {}
            }
            
            # For each non-Bayesian principal, find cases with maximum difference
            for principal_type in model_results:
                if principal_type != 'bayesian':
                    principal_case_details = model_results[principal_type]['case_details']
                    
                    # Calculate differences for each case
                    case_differences = {}
                    for case_id in principal_case_details:
                        if case_id in bayesian_case_details:
                            principal_rate = principal_case_details[case_id]['acceptance_rate']
                            bayesian_rate = bayesian_case_details[case_id]['acceptance_rate']
                            difference = abs(principal_rate - bayesian_rate)
                            case_differences[case_id] = {
                                'difference': difference,
                                'principal_rate': principal_rate,
                                'bayesian_rate': bayesian_rate,
                                'principal_accepts': principal_case_details[case_id]['accepts'],
                                'principal_total': principal_case_details[case_id]['total'],
                                'bayesian_accepts': bayesian_case_details[case_id]['accepts'],
                                'bayesian_total': bayesian_case_details[case_id]['total']
                            }
                    
                    # Sort by difference and take top 3
                    sorted_cases = sorted(case_differences.items(), key=lambda x: x[1]['difference'], reverse=True)
                    top_3_cases = sorted_cases[:3]
                    
                    # Store top 3 cases for this principal
                    model_max_diff_data['principals'][principal_type] = [
                        {
                            'case_id': case_id,
                            **case_info
                        }
                        for case_id, case_info in top_3_cases
                    ]
            
            # Save one file per model with all principals' top 3 cases
            output_file = f'{args.output}_{model_name}_max_diff.json'
            with open(output_file, 'w') as f:
                json.dump(model_max_diff_data, f, indent=2)
            
            print(f"Saved max difference cases for {model_name}: {output_file}")
    
    # Create plots for each model
    for model_name, model_results in all_results.items():
        # Create two subplots: one for fine acceptance rate, one for coarse acceptance rate
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Define the desired order (abbreviated labels)
        desired_order = ["bayesian", "anchoring_principal", "availability_principal", "confirmation_principal", "conservatism_principal", "overconfidence_principal", "prospect_principal"]
        
        # Filter to only include principal types that exist in the data
        principal_types = [pt for pt in desired_order if pt in model_results]
        acceptance_rates = [model_results[pt]['acceptance_rate'] for pt in principal_types]
        coarse_acceptance_rates = [model_results[pt]['coarse_acceptance_rate'] for pt in principal_types]
        display_labels = [pt.replace('_principal', '') for pt in principal_types]
        
        # Create color array
        colors = []
        for pt in principal_types:
            if pt == 'bayesian':
                colors.append('blue')
            else:
                colors.append('gray')
        
        # Plot 1: Fine acceptance rate
        bars1 = ax1.bar(range(len(principal_types)), acceptance_rates, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars1, acceptance_rates)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1%}',
                   ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Principal Type', fontsize=12)
        ax1.set_ylabel('Fine Acceptance Rate', fontsize=12)
        ax1.set_title(f'Fine Acceptance Rates - {model_name}\n(Blue = Bayesian Baseline)', 
                    fontsize=14)
        ax1.set_xticks(range(len(principal_types)))
        ax1.set_xticklabels(display_labels, rotation=45, ha='right')
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Coarse acceptance rate
        bars2 = ax2.bar(range(len(principal_types)), coarse_acceptance_rates, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars2, coarse_acceptance_rates)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1%}',
                   ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('Principal Type', fontsize=12)
        ax2.set_ylabel('Coarse Acceptance Rate', fontsize=12)
        ax2.set_title(f'Coarse Acceptance Rates - {model_name}\n(Threshold: {args.coarse_threshold}, Blue = Bayesian Baseline)', 
                    fontsize=14)
        ax2.set_xticks(range(len(principal_types)))
        ax2.set_xticklabels(display_labels, rotation=45, ha='right')
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{args.output}_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved plot: {args.output}_{model_name}.png")
    
    # Save all results to JSON
    output_data = {
        'results': all_results,
        'coarse_threshold': args.coarse_threshold
    }
    
    with open(f'{args.output}.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved results: {args.output}.json")