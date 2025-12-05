import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_belief_data(belief_file):
    """Load belief data from JSON file"""
    with open(belief_file, 'r') as f:
        data = json.load(f)

    belief_dict = {}
    for result in data.get('results', []):
        case_id = result.get('id')
        belief = result.get('belief')
        correct = result.get('correct', False)
        if case_id and belief is not None:
            belief_dict[case_id] = {
                'belief': belief,
                'correct': correct,
                'predicted_answer': result.get('predicted_answer_idx'),
                'correct_answer': result.get('correct_answer_idx')
            }
    return belief_dict, data.get('metrics', {})

def load_principal_data(input_dir, model_name):
    """Load bayesian and behavioral principal data for a specific model"""
    bayesian_file = os.path.join(input_dir, f'principal_{model_name}_bayesian.json')
    behavioral_file = os.path.join(input_dir, f'principal_{model_name}_behavioral.json')

    data = {}

    for filepath, principal_type in [(bayesian_file, 'bayesian'), (behavioral_file, 'behavioral')]:
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r') as f:
            raw_data = json.load(f)

        for item in raw_data:
            case_id = item['case_id']
            decision = item.get('decision', '').strip().lower()
            is_accept = 'accept' in decision

            if case_id not in data:
                data[case_id] = {}

            data[case_id][principal_type] = {
                'decision': 'accept' if is_accept else 'reject'
            }

    return data

def generate_belief_statistics(belief_data):
    """Generate statistics about belief distribution"""
    beliefs = [item['belief'] for item in belief_data.values()]
    correct_beliefs = [item['belief'] for item in belief_data.values() if item['correct']]
    incorrect_beliefs = [item['belief'] for item in belief_data.values() if not item['correct']]

    stats = {
        'total_cases': len(beliefs),
        'mean_belief': np.mean(beliefs),
        'median_belief': np.median(beliefs),
        'std_belief': np.std(beliefs),
        'min_belief': np.min(beliefs),
        'max_belief': np.max(beliefs),
        'correct_cases': len(correct_beliefs),
        'incorrect_cases': len(incorrect_beliefs),
        'mean_belief_correct': np.mean(correct_beliefs) if correct_beliefs else 0,
        'mean_belief_incorrect': np.mean(incorrect_beliefs) if incorrect_beliefs else 0,
    }

    # Belief quantiles
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    for q in quantiles:
        stats[f'quantile_{int(q*100)}'] = np.quantile(beliefs, q)

    return stats, beliefs, correct_beliefs, incorrect_beliefs

def plot_belief_distribution(beliefs, correct_beliefs, incorrect_beliefs, output_dir):
    """Create belief distribution plots"""
    plt.style.use('seaborn-v0_8-darkgrid')

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Overall belief distribution
    ax1 = axes[0, 0]
    ax1.hist(beliefs, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(np.mean(beliefs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(beliefs):.3f}')
    ax1.set_xlabel('Belief Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Overall Belief Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Correct vs Incorrect belief distribution
    ax2 = axes[0, 1]
    ax2.hist(correct_beliefs, bins=30, alpha=0.6, color='green', label='Correct', edgecolor='black')
    ax2.hist(incorrect_beliefs, bins=30, alpha=0.6, color='red', label='Incorrect', edgecolor='black')
    ax2.set_xlabel('Belief Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Belief Distribution: Correct vs Incorrect', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Box plot comparison
    ax3 = axes[1, 0]
    box_data = [correct_beliefs, incorrect_beliefs]
    bp = ax3.boxplot(box_data, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Belief Score', fontsize=12)
    ax3.set_title('Belief Score Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Cumulative distribution
    ax4 = axes[1, 1]
    sorted_beliefs = np.sort(beliefs)
    cumulative = np.arange(1, len(sorted_beliefs) + 1) / len(sorted_beliefs)
    ax4.plot(sorted_beliefs, cumulative, linewidth=2, color='steelblue')
    ax4.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    ax4.axhline(0.8, color='orange', linestyle='--', alpha=0.5, label='80th percentile')
    ax4.set_xlabel('Belief Score', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'belief_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'belief_distribution.png')}")
    plt.close()

def plot_principal_comparison(all_models_data, output_dir):
    """Create comparison plots for bayesian vs behavioral principals"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = list(all_models_data.keys())
    bayesian_accept_rates = []
    behavioral_accept_rates = []
    disagreement_rates = []

    for model in models:
        data = all_models_data[model]
        bayesian_accept_rates.append(data['bayesian_accept_rate'])
        behavioral_accept_rates.append(data['behavioral_accept_rate'])
        disagreement_rates.append(data['disagreement_rate'])

    # Plot 1: Acceptance rates comparison
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x - width/2, bayesian_accept_rates, width, label='Bayesian', color='steelblue', edgecolor='black')
    ax1.bar(x + width/2, behavioral_accept_rates, width, label='Behavioral', color='coral', edgecolor='black')
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Acceptance Rate', fontsize=12)
    ax1.set_title('Acceptance Rates: Bayesian vs Behavioral', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Disagreement rates
    ax2 = axes[0, 1]
    ax2.bar(models, disagreement_rates, color='purple', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Disagreement Rate', fontsize=12)
    ax2.set_title('Principal Disagreement Rate by Model', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Scatter plot of acceptance rates
    ax3 = axes[1, 0]
    ax3.scatter(bayesian_accept_rates, behavioral_accept_rates, s=100, alpha=0.6, color='green', edgecolor='black')
    for i, model in enumerate(models):
        ax3.annotate(model, (bayesian_accept_rates[i], behavioral_accept_rates[i]),
                    fontsize=8, xytext=(5, 5), textcoords='offset points')
    ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Equal rates')
    ax3.set_xlabel('Bayesian Acceptance Rate', fontsize=12)
    ax3.set_ylabel('Behavioral Acceptance Rate', fontsize=12)
    ax3.set_title('Bayesian vs Behavioral Acceptance Rates', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Difference in acceptance rates
    ax4 = axes[1, 1]
    differences = [b - beh for b, beh in zip(bayesian_accept_rates, behavioral_accept_rates)]
    colors = ['green' if d > 0 else 'red' for d in differences]
    ax4.barh(models, differences, color=colors, alpha=0.7, edgecolor='black')
    ax4.axvline(0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Difference (Bayesian - Behavioral)', fontsize=12)
    ax4.set_ylabel('Model', fontsize=12)
    ax4.set_title('Acceptance Rate Difference', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'principal_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'principal_comparison.png')}")
    plt.close()

def plot_belief_vs_disagreement(all_models_data, belief_data, output_dir):
    """Plot relationship between belief and principal disagreement"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Collect all disagreement cases with beliefs
    all_disagree_beliefs = []
    all_agree_beliefs = []

    for model_name, model_data in all_models_data.items():
        for case in model_data.get('cases', []):
            case_id = case['case_id']
            if case_id in belief_data:
                belief = belief_data[case_id]['belief']
                if case['bayesian_decision'] != case['behavioral_decision']:
                    all_disagree_beliefs.append(belief)
                else:
                    all_agree_beliefs.append(belief)

    # Plot 1: Belief distribution for agreement vs disagreement
    ax1 = axes[0]
    ax1.hist(all_agree_beliefs, bins=30, alpha=0.6, color='green', label='Agreement', edgecolor='black')
    ax1.hist(all_disagree_beliefs, bins=30, alpha=0.6, color='red', label='Disagreement', edgecolor='black')
    ax1.set_xlabel('Belief Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Belief Distribution: Agreement vs Disagreement', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Box plot
    ax2 = axes[1]
    box_data = [all_agree_beliefs, all_disagree_beliefs]
    bp = ax2.boxplot(box_data, labels=['Agreement', 'Disagreement'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Belief Score', fontsize=12)
    ax2.set_title('Belief Score: Agreement vs Disagreement', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics
    if all_agree_beliefs and all_disagree_beliefs:
        mean_agree = np.mean(all_agree_beliefs)
        mean_disagree = np.mean(all_disagree_beliefs)
        ax2.text(0.05, 0.95, f'Mean (Agreement): {mean_agree:.3f}\nMean (Disagreement): {mean_disagree:.3f}',
                transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'belief_vs_disagreement.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'belief_vs_disagreement.png')}")
    plt.close()

def generate_summary_table(all_models_data, belief_stats):
    """Generate summary table as markdown"""
    table = "# Decision-Making Analysis Summary\n\n"

    # Belief statistics table
    table += "## DeepSeek Belief Statistics\n\n"
    table += "| Metric | Value |\n"
    table += "|--------|-------|\n"
    table += f"| Total Cases | {belief_stats['total_cases']} |\n"
    table += f"| Mean Belief | {belief_stats['mean_belief']:.4f} |\n"
    table += f"| Median Belief | {belief_stats['median_belief']:.4f} |\n"
    table += f"| Std Deviation | {belief_stats['std_belief']:.4f} |\n"
    table += f"| Min Belief | {belief_stats['min_belief']:.4f} |\n"
    table += f"| Max Belief | {belief_stats['max_belief']:.4f} |\n"
    table += f"| Correct Cases | {belief_stats['correct_cases']} ({belief_stats['correct_cases']/belief_stats['total_cases']*100:.1f}%) |\n"
    table += f"| Incorrect Cases | {belief_stats['incorrect_cases']} ({belief_stats['incorrect_cases']/belief_stats['total_cases']*100:.1f}%) |\n"
    table += f"| Mean Belief (Correct) | {belief_stats['mean_belief_correct']:.4f} |\n"
    table += f"| Mean Belief (Incorrect) | {belief_stats['mean_belief_incorrect']:.4f} |\n"

    table += "\n### Belief Quantiles\n\n"
    table += "| Quantile | Value |\n"
    table += "|----------|-------|\n"
    for q in [10, 25, 50, 75, 90]:
        table += f"| {q}th percentile | {belief_stats[f'quantile_{q}']:.4f} |\n"

    # Principal comparison table
    table += "\n## Principal Decision-Making Comparison\n\n"
    table += "| Model | Bayesian Accept Rate | Behavioral Accept Rate | Difference | Disagreement Rate | Total Cases |\n"
    table += "|-------|---------------------|----------------------|------------|-------------------|-------------|\n"

    for model_name in sorted(all_models_data.keys()):
        data = all_models_data[model_name]
        bayesian_rate = data['bayesian_accept_rate']
        behavioral_rate = data['behavioral_accept_rate']
        diff = bayesian_rate - behavioral_rate
        disagree_rate = data['disagreement_rate']
        total = data['total_cases']

        table += f"| {model_name} | {bayesian_rate:.2%} | {behavioral_rate:.2%} | {diff:+.2%} | {disagree_rate:.2%} | {total} |\n"

    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate statistics and visualizations for USMLE sample analysis.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='output/usmle_sample',
        help='Directory containing USMLE sample principal JSON files'
    )
    parser.add_argument(
        '--belief-file',
        type=str,
        default='tests/test_usmle_sample_deepseek-ai-deepseek-v3.1_belief.json',
        help='Path to belief file from DeepSeek-v3.1 model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis/usmle_sample',
        help='Directory for saving analysis outputs'
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load belief data
    print(f"Loading belief data from: {args.belief_file}")
    belief_data, belief_metrics = load_belief_data(args.belief_file)
    print(f"  Loaded beliefs for {len(belief_data)} cases")

    # Generate belief statistics
    print("\nGenerating belief statistics...")
    belief_stats, beliefs, correct_beliefs, incorrect_beliefs = generate_belief_statistics(belief_data)

    # Plot belief distributions
    print("Creating belief distribution plots...")
    plot_belief_distribution(beliefs, correct_beliefs, incorrect_beliefs, args.output_dir)

    # Load principal data for all models
    print("\nLoading principal decision data...")
    all_models_data = {}

    models = ['deepseek', 'llama', 'llama-large', 'llama-dpo', 'llama-sft', 'llama-small']

    for model_name in models:
        print(f"  Processing {model_name}...")
        principal_data = load_principal_data(args.input_dir, model_name)

        if not principal_data:
            print(f"    No data found for {model_name}")
            continue

        # Calculate statistics
        total_cases = 0
        bayesian_accepts = 0
        behavioral_accepts = 0
        disagreements = 0
        cases = []

        for case_id, decisions in principal_data.items():
            if 'bayesian' in decisions and 'behavioral' in decisions:
                total_cases += 1

                bayesian_decision = decisions['bayesian']['decision']
                behavioral_decision = decisions['behavioral']['decision']

                if bayesian_decision == 'accept':
                    bayesian_accepts += 1
                if behavioral_decision == 'accept':
                    behavioral_accepts += 1
                if bayesian_decision != behavioral_decision:
                    disagreements += 1

                cases.append({
                    'case_id': case_id,
                    'bayesian_decision': bayesian_decision,
                    'behavioral_decision': behavioral_decision
                })

        if total_cases > 0:
            all_models_data[model_name] = {
                'total_cases': total_cases,
                'bayesian_accept_rate': bayesian_accepts / total_cases,
                'behavioral_accept_rate': behavioral_accepts / total_cases,
                'disagreement_rate': disagreements / total_cases,
                'disagreements': disagreements,
                'cases': cases
            }

    # Plot principal comparisons
    print("\nCreating principal comparison plots...")
    plot_principal_comparison(all_models_data, args.output_dir)

    # Plot belief vs disagreement
    print("Creating belief vs disagreement plots...")
    plot_belief_vs_disagreement(all_models_data, belief_data, args.output_dir)

    # Generate summary table
    print("\nGenerating summary table...")
    summary_table = generate_summary_table(all_models_data, belief_stats)

    summary_file = os.path.join(args.output_dir, 'summary.md')
    with open(summary_file, 'w') as f:
        f.write(summary_table)
    print(f"Saved: {summary_file}")

    # Save detailed statistics as JSON
    detailed_stats = {
        'belief_statistics': belief_stats,
        'belief_metrics': belief_metrics,
        'model_comparisons': all_models_data
    }

    stats_file = os.path.join(args.output_dir, 'detailed_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(detailed_stats, f, indent=2)
    print(f"Saved: {stats_file}")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print(f"\nOutputs saved to: {args.output_dir}")
    print("  - belief_distribution.png")
    print("  - principal_comparison.png")
    print("  - belief_vs_disagreement.png")
    print("  - summary.md")
    print("  - detailed_statistics.json")
