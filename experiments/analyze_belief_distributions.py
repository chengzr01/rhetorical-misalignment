#!/usr/bin/env python3
"""
Analyze and compare belief distributions of Bayesian and behavioral principals
across different models (llama-small, llama-dpo, llama-sft).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_belief_data(file_path: str) -> List[Dict]:
    """Load belief data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def extract_beliefs_and_accuracy(data: List[Dict]) -> Tuple[List[float], List[bool]]:
    """
    Extract belief values and correctness from data.

    Returns:
        beliefs: List of belief values
        correct: List of boolean indicating if decision was correct
    """
    beliefs = []
    correct = []

    for case in data:
        try:
            belief = float(case['belief'])
            beliefs.append(belief)

            decision = case['decision']
            correct_answer = case['correct_answer_idx']
            correct.append(decision == correct_answer)
        except (ValueError, KeyError) as e:
            print(f"Warning: Skipping case {case.get('case_id', 'unknown')} due to: {e}")
            continue

    return beliefs, correct


def compute_statistics(beliefs: List[float], correct: List[bool]) -> Dict:
    """Compute comprehensive statistics for beliefs."""
    beliefs_arr = np.array(beliefs)
    correct_arr = np.array(correct)

    # Overall statistics
    stats_dict = {
        'count': len(beliefs),
        'mean': np.mean(beliefs_arr),
        'median': np.median(beliefs_arr),
        'std': np.std(beliefs_arr),
        'min': np.min(beliefs_arr),
        'max': np.max(beliefs_arr),
        'q25': np.percentile(beliefs_arr, 25),
        'q75': np.percentile(beliefs_arr, 75),
        'accuracy': np.mean(correct_arr),
    }

    # Beliefs when correct vs incorrect
    beliefs_correct = beliefs_arr[correct_arr]
    beliefs_incorrect = beliefs_arr[~correct_arr]

    if len(beliefs_correct) > 0:
        stats_dict['mean_belief_correct'] = np.mean(beliefs_correct)
        stats_dict['median_belief_correct'] = np.median(beliefs_correct)
    else:
        stats_dict['mean_belief_correct'] = np.nan
        stats_dict['median_belief_correct'] = np.nan

    if len(beliefs_incorrect) > 0:
        stats_dict['mean_belief_incorrect'] = np.mean(beliefs_incorrect)
        stats_dict['median_belief_incorrect'] = np.median(beliefs_incorrect)
    else:
        stats_dict['mean_belief_incorrect'] = np.nan
        stats_dict['median_belief_incorrect'] = np.nan

    # Calibration: Check if higher belief correlates with being correct
    if len(beliefs) > 1:
        # Point-biserial correlation between belief and correctness
        correlation, p_value = stats.pointbiserialr(correct_arr, beliefs_arr)
        stats_dict['belief_correctness_correlation'] = correlation
        stats_dict['belief_correctness_p_value'] = p_value
    else:
        stats_dict['belief_correctness_correlation'] = np.nan
        stats_dict['belief_correctness_p_value'] = np.nan

    return stats_dict


def compare_models_and_principals(output_dir: str = 'principals/usmle_sample') -> pd.DataFrame:
    """
    Load and compare all models and principal types.

    Returns:
        DataFrame with statistics for each model-principal combination
    """
    models = ['llama-small', 'llama-dpo', 'llama-sft']
    principal_types = ['bayesian', 'behavioral']

    results = []

    for model in models:
        for principal_type in principal_types:
            file_path = f"{output_dir}/principal_{model}_{principal_type}_belief.json"

            try:
                data = load_belief_data(file_path)
                beliefs, correct = extract_beliefs_and_accuracy(data)
                stats_dict = compute_statistics(beliefs, correct)

                stats_dict['model'] = model
                stats_dict['principal_type'] = principal_type
                results.append(stats_dict)

                print(f"Loaded {model} - {principal_type}: {len(beliefs)} cases")
            except FileNotFoundError:
                print(f"Warning: File not found: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    df = pd.DataFrame(results)
    return df


def plot_belief_distributions(output_dir: str = 'principals/usmle_sample',
                              save_path: str = 'analysis/belief_distribution_comparison.png'):
    """Plot belief distributions for all model-principal combinations."""
    models = ['llama-small', 'llama-dpo', 'llama-sft']
    principal_types = ['bayesian', 'behavioral']

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Belief Distributions: Bayesian vs Behavioral Principals', fontsize=16)

    for i, model in enumerate(models):
        for j, principal_type in enumerate(principal_types):
            ax = axes[i, j]
            file_path = f"{output_dir}/principal_{model}_{principal_type}_belief.json"

            try:
                data = load_belief_data(file_path)
                beliefs, correct = extract_beliefs_and_accuracy(data)

                # Separate beliefs by correctness
                beliefs_arr = np.array(beliefs)
                correct_arr = np.array(correct)

                beliefs_correct = beliefs_arr[correct_arr]
                beliefs_incorrect = beliefs_arr[~correct_arr]

                # Plot histograms
                ax.hist(beliefs_correct, bins=20, alpha=0.6, label='Correct', color='green', density=True)
                ax.hist(beliefs_incorrect, bins=20, alpha=0.6, label='Incorrect', color='red', density=True)

                ax.set_xlabel('Belief', fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.set_title(f'{model} - {principal_type}\nAcc: {np.mean(correct_arr):.3f}, Mean belief: {np.mean(beliefs_arr):.3f}',
                           fontsize=11)
                ax.legend()
                ax.grid(True, alpha=0.3)

            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model} - {principal_type}')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def plot_comparison_by_principal_type(output_dir: str = 'principals/usmle_sample',
                                     save_path: str = 'analysis/principal_type_comparison.png'):
    """Plot side-by-side comparison of Bayesian vs Behavioral for each model."""
    models = ['llama-small', 'llama-dpo', 'llama-sft']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Bayesian vs Behavioral Principal Comparison', fontsize=16)

    for i, model in enumerate(models):
        ax = axes[i]

        bayesian_file = f"{output_dir}/principal_{model}_bayesian_belief.json"
        behavioral_file = f"{output_dir}/principal_{model}_behavioral_belief.json"

        try:
            # Load Bayesian data
            bayesian_data = load_belief_data(bayesian_file)
            bayesian_beliefs, bayesian_correct = extract_beliefs_and_accuracy(bayesian_data)

            # Load Behavioral data
            behavioral_data = load_belief_data(behavioral_file)
            behavioral_beliefs, behavioral_correct = extract_beliefs_and_accuracy(behavioral_data)

            # Create violin plots
            data_to_plot = [bayesian_beliefs, behavioral_beliefs]
            positions = [1, 2]

            parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)

            # Customize colors
            for i_part, pc in enumerate(parts['bodies']):
                if i_part == 0:
                    pc.set_facecolor('skyblue')
                else:
                    pc.set_facecolor('lightcoral')
                pc.set_alpha(0.7)

            ax.set_xticks(positions)
            ax.set_xticklabels(['Bayesian', 'Behavioral'])
            ax.set_ylabel('Belief', fontsize=11)
            ax.set_title(f'{model}\nBay Acc: {np.mean(bayesian_correct):.3f}, Beh Acc: {np.mean(behavioral_correct):.3f}',
                       fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

            # Add mean values as text
            ax.text(1, np.mean(bayesian_beliefs) + 0.02, f'{np.mean(bayesian_beliefs):.3f}',
                   ha='center', fontsize=9)
            ax.text(2, np.mean(behavioral_beliefs) + 0.02, f'{np.mean(behavioral_beliefs):.3f}',
                   ha='center', fontsize=9)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{model}')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


def plot_calibration_curves(output_dir: str = 'principals/usmle_sample',
                            save_path: str = 'analysis/calibration_curves.png'):
    """Plot calibration curves showing belief vs actual accuracy."""
    models = ['llama-small', 'llama-dpo', 'llama-sft']
    principal_types = ['bayesian', 'behavioral']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Calibration: Belief vs Actual Accuracy', fontsize=16)

    for j, principal_type in enumerate(principal_types):
        ax = axes[j]

        for model in models:
            file_path = f"{output_dir}/principal_{model}_{principal_type}_belief.json"

            try:
                data = load_belief_data(file_path)
                beliefs, correct = extract_beliefs_and_accuracy(data)

                # Bin beliefs and compute accuracy per bin
                beliefs_arr = np.array(beliefs)
                correct_arr = np.array(correct).astype(float)

                # Create bins
                bins = np.linspace(0, 1, 11)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                bin_accuracies = []
                bin_counts = []

                for i in range(len(bins) - 1):
                    mask = (beliefs_arr >= bins[i]) & (beliefs_arr < bins[i+1])
                    if np.sum(mask) > 0:
                        bin_accuracies.append(np.mean(correct_arr[mask]))
                        bin_counts.append(np.sum(mask))
                    else:
                        bin_accuracies.append(np.nan)
                        bin_counts.append(0)

                # Plot with marker size proportional to count
                sizes = np.array(bin_counts) * 2
                ax.plot(bin_centers, bin_accuracies, marker='o', label=model, linewidth=2, markersize=6)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Perfect calibration')

        ax.set_xlabel('Belief', fontsize=11)
        ax.set_ylabel('Actual Accuracy', fontsize=11)
        ax.set_title(f'{principal_type.capitalize()} Principal', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


def analyze_belief_differences(output_dir: str = 'principals/usmle_sample',
                               save_path: str = 'analysis/belief_differences.txt'):
    """Analyze case-by-case differences between Bayesian and behavioral beliefs."""
    models = ['llama-small', 'llama-dpo', 'llama-sft']

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("CASE-BY-CASE BELIEF DIFFERENCES: BAYESIAN vs BEHAVIORAL")
    output_lines.append("=" * 80)
    output_lines.append("")

    for model in models:
        bayesian_file = f"{output_dir}/principal_{model}_bayesian_belief.json"
        behavioral_file = f"{output_dir}/principal_{model}_behavioral_belief.json"

        try:
            bayesian_data = load_belief_data(bayesian_file)
            behavioral_data = load_belief_data(behavioral_file)

            # Match cases by case_id
            bayesian_dict = {case['case_id']: case for case in bayesian_data}
            behavioral_dict = {case['case_id']: case for case in behavioral_data}

            common_cases = set(bayesian_dict.keys()) & set(behavioral_dict.keys())

            differences = []
            for case_id in common_cases:
                bay_case = bayesian_dict[case_id]
                beh_case = behavioral_dict[case_id]

                try:
                    bay_belief = float(bay_case['belief'])
                    beh_belief = float(beh_case['belief'])
                    diff = bay_belief - beh_belief

                    differences.append({
                        'case_id': case_id,
                        'bayesian_belief': bay_belief,
                        'behavioral_belief': beh_belief,
                        'difference': diff,
                        'abs_difference': abs(diff),
                        'bayesian_decision': bay_case['decision'],
                        'behavioral_decision': beh_case['decision'],
                        'correct_answer': bay_case['correct_answer_idx'],
                        'decisions_match': bay_case['decision'] == beh_case['decision'],
                    })
                except (ValueError, KeyError):
                    continue

            if differences:
                df_diff = pd.DataFrame(differences)

                output_lines.append(f"\n{model.upper()}")
                output_lines.append("-" * 80)
                output_lines.append(f"Number of cases: {len(differences)}")
                output_lines.append(f"Mean belief difference (Bayesian - Behavioral): {df_diff['difference'].mean():.4f}")
                output_lines.append(f"Std of belief difference: {df_diff['difference'].std():.4f}")
                output_lines.append(f"Mean absolute difference: {df_diff['abs_difference'].mean():.4f}")
                output_lines.append(f"Max absolute difference: {df_diff['abs_difference'].max():.4f}")
                output_lines.append(f"Cases where decisions differ: {(~df_diff['decisions_match']).sum()} ({(~df_diff['decisions_match']).sum()/len(differences)*100:.1f}%)")

                # Statistical test
                t_stat, p_value = stats.ttest_rel(df_diff['bayesian_belief'], df_diff['behavioral_belief'])
                output_lines.append(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")

                # Top 5 cases with largest differences
                output_lines.append(f"\nTop 5 cases with largest absolute differences:")
                top_cases = df_diff.nlargest(5, 'abs_difference')
                for idx, row in top_cases.iterrows():
                    output_lines.append(f"  Case {row['case_id']}: Bay={row['bayesian_belief']:.3f}, Beh={row['behavioral_belief']:.3f}, Diff={row['difference']:+.3f}")
                    output_lines.append(f"    Decisions: Bay={row['bayesian_decision']}, Beh={row['behavioral_decision']}, Correct={row['correct_answer']}, Match={row['decisions_match']}")

        except Exception as e:
            output_lines.append(f"\nError processing {model}: {e}")

    # Save to file
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"\nBelief differences analysis saved to: {save_path}")

    # Also print to console
    print('\n'.join(output_lines))


def main():
    """Main analysis function."""
    print("=" * 80)
    print("BELIEF DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()

    # Compare models and principals
    print("Computing statistics...")
    df_stats = compare_models_and_principals()

    # Save statistics table
    stats_file = 'analysis/belief_statistics_summary.csv'
    Path(stats_file).parent.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(stats_file, index=False)
    print(f"\nStatistics saved to: {stats_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Format for display
    display_cols = ['model', 'principal_type', 'count', 'mean', 'median', 'std',
                   'accuracy', 'mean_belief_correct', 'mean_belief_incorrect',
                   'belief_correctness_correlation']

    print("\n" + df_stats[display_cols].to_string(index=False))

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    plot_belief_distributions()
    plot_comparison_by_principal_type()
    plot_calibration_curves()

    # Analyze belief differences
    print("\n" + "=" * 80)
    print("ANALYZING BELIEF DIFFERENCES")
    print("=" * 80)
    analyze_belief_differences()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - analysis/belief_statistics_summary.csv")
    print("  - analysis/belief_distribution_comparison.png")
    print("  - analysis/principal_type_comparison.png")
    print("  - analysis/calibration_curves.png")
    print("  - analysis/belief_differences.txt")


if __name__ == "__main__":
    main()
