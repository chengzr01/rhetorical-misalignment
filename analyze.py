"""
Analyze results from principal-agent simulation experiments.

This script analyzes decision distributions, belief statistics, and persuasion
effects across different agent and principal types in clinical decision-making.

Example usage:
    # Analyze all results files in a folder
    python analyze.py --input experiments/output
    
    # Analyze results and save to custom output directory
    python analyze.py \
        --input experiments/output \
        --output-dir experiments/analysis
        
    # Analyze results without generating plots
    python analyze.py \
        --input experiments/output \
        --no-plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_results_from_folder(folder: str | Path) -> list[Mapping[str, Any]]:
    """Load and combine experiment results from all JSON files in a folder."""
    folder_path = Path(folder)
    all_results = []
    
    # Get all JSON files in the folder
    json_files = list(folder_path.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in folder: {folder}")
        
    for filepath in json_files:
        print(f"Loading results from: {filepath}")
        with open(filepath, "r") as f:
            results = json.load(f)
            all_results.extend(results)
            
    return all_results


def compute_statistics(results: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    """
    Compute comprehensive statistics from experiment results.

    Returns:
        Dictionary containing:
        - overall_stats: Overall decision distribution
        - by_agent: Statistics grouped by agent type
        - by_principal: Statistics grouped by principal type
        - by_pair: Statistics for each agent-principal pair
        - belief_stats: Belief distribution statistics
    """
    stats = {
        "overall_stats": {},
        "by_agent": defaultdict(dict),
        "by_principal": defaultdict(dict),
        "by_pair": defaultdict(dict),
        "belief_stats": {},
    }

    # Overall statistics
    total = len(results)
    treat_count = sum(1 for r in results if r["decision"] == "treat")

    stats["overall_stats"] = {
        "total_decisions": total,
        "treat_count": treat_count,
        "do_not_treat_count": total - treat_count,
        "treat_rate": treat_count / total if total > 0 else 0,
    }

    # Group by agent type
    by_agent = defaultdict(list)
    for r in results:
        by_agent[r["agent_name"]].append(r)

    for agent_name, agent_results in by_agent.items():
        total_agent = len(agent_results)
        treat_agent = sum(1 for r in agent_results if r["decision"] == "treat")
        beliefs = [float(r["belief"]) for r in agent_results if r["belief"]]

        stats["by_agent"][agent_name] = {
            "total_decisions": total_agent,
            "treat_count": treat_agent,
            "do_not_treat_count": total_agent - treat_agent,
            "treat_rate": treat_agent / total_agent if total_agent > 0 else 0,
            "avg_belief": float(np.mean(beliefs)) if beliefs else 0,
            "median_belief": float(np.median(beliefs)) if beliefs else 0,
            "std_belief": float(np.std(beliefs)) if beliefs else 0,
        }

    # Group by principal type
    by_principal = defaultdict(list)
    for r in results:
        by_principal[r["principal_name"]].append(r)

    for principal_name, principal_results in by_principal.items():
        total_principal = len(principal_results)
        treat_principal = sum(1 for r in principal_results if r["decision"] == "treat")
        beliefs = [float(r["belief"]) for r in principal_results if r["belief"]]

        stats["by_principal"][principal_name] = {
            "total_decisions": total_principal,
            "treat_count": treat_principal,
            "do_not_treat_count": total_principal - treat_principal,
            "treat_rate": treat_principal / total_principal if total_principal > 0 else 0,
            "avg_belief": float(np.mean(beliefs)) if beliefs else 0,
            "median_belief": float(np.median(beliefs)) if beliefs else 0,
            "std_belief": float(np.std(beliefs)) if beliefs else 0,
        }

    # Group by agent-principal pair
    by_pair = defaultdict(list)
    for r in results:
        pair_key = f"{r['agent_name']}_{r['principal_name']}"
        by_pair[pair_key].append(r)

    for pair_key, pair_results in by_pair.items():
        total_pair = len(pair_results)
        treat_pair = sum(1 for r in pair_results if r["decision"] == "treat")
        beliefs = [float(r["belief"]) for r in pair_results if r["belief"]]

        stats["by_pair"][pair_key] = {
            "agent_name": pair_results[0]["agent_name"],
            "principal_name": pair_results[0]["principal_name"],
            "total_decisions": total_pair,
            "treat_count": treat_pair,
            "do_not_treat_count": total_pair - treat_pair,
            "treat_rate": treat_pair / total_pair if total_pair > 0 else 0,
            "avg_belief": float(np.mean(beliefs)) if beliefs else 0,
            "median_belief": float(np.median(beliefs)) if beliefs else 0,
            "std_belief": float(np.std(beliefs)) if beliefs else 0,
            "min_belief": float(np.min(beliefs)) if beliefs else 0,
            "max_belief": float(np.max(beliefs)) if beliefs else 0,
        }

    # Belief statistics
    all_beliefs = [float(r["belief"]) for r in results if r["belief"]]
    if all_beliefs:
        stats["belief_stats"] = {
            "mean": float(np.mean(all_beliefs)),
            "median": float(np.median(all_beliefs)),
            "std": float(np.std(all_beliefs)),
            "min": float(np.min(all_beliefs)),
            "max": float(np.max(all_beliefs)),
            "percentile_25": float(np.percentile(all_beliefs, 25)),
            "percentile_75": float(np.percentile(all_beliefs, 75)),
        }

    # Convert defaultdicts to regular dicts for JSON serialization
    stats["by_agent"] = dict(stats["by_agent"])
    stats["by_principal"] = dict(stats["by_principal"])
    stats["by_pair"] = dict(stats["by_pair"])

    return stats


def plot_decision_distribution(
    results: list[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    """Plot decision distribution by agent and principal types."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Decision Distribution Analysis", fontsize=16, fontweight='bold')

    # 1. Overall decision distribution
    decisions = [r["decision"] for r in results]
    decision_counts = {"treat": decisions.count("treat"), "do not treat": decisions.count("do not treat")}

    axes[0, 0].bar(decision_counts.keys(), decision_counts.values(), color=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title("Overall Decision Distribution", fontweight='bold')
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, (k, v) in enumerate(decision_counts.items()):
        axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

    # 2. Decision distribution by agent type
    by_agent = defaultdict(lambda: {"treat": 0, "do not treat": 0})
    for r in results:
        by_agent[r["agent_name"]][r["decision"]] += 1

    agent_names = list(by_agent.keys())
    treat_counts = [by_agent[a]["treat"] for a in agent_names]
    no_treat_counts = [by_agent[a]["do not treat"] for a in agent_names]

    x = np.arange(len(agent_names))
    width = 0.35
    axes[0, 1].bar(x - width/2, treat_counts, width, label='Treat', color='#2ecc71')
    axes[0, 1].bar(x + width/2, no_treat_counts, width, label='Do Not Treat', color='#e74c3c')
    axes[0, 1].set_title("Decisions by Agent Type", fontweight='bold')
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(agent_names, rotation=15, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. Decision distribution by principal type
    by_principal = defaultdict(lambda: {"treat": 0, "do not treat": 0})
    for r in results:
        by_principal[r["principal_name"]][r["decision"]] += 1

    principal_names = list(by_principal.keys())
    treat_counts = [by_principal[p]["treat"] for p in principal_names]
    no_treat_counts = [by_principal[p]["do not treat"] for p in principal_names]

    x = np.arange(len(principal_names))
    axes[1, 0].bar(x - width/2, treat_counts, width, label='Treat', color='#2ecc71')
    axes[1, 0].bar(x + width/2, no_treat_counts, width, label='Do Not Treat', color='#e74c3c')
    axes[1, 0].set_title("Decisions by Principal Type", fontweight='bold')
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(principal_names, rotation=15, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. Treat rate by agent-principal pairs
    by_pair = defaultdict(lambda: {"treat": 0, "total": 0})
    for r in results:
        pair_key = f"{r['agent_name']}\n{r['principal_name']}"
        by_pair[pair_key]["total"] += 1
        if r["decision"] == "treat":
            by_pair[pair_key]["treat"] += 1

    pairs = list(by_pair.keys())
    treat_rates = [by_pair[p]["treat"] / by_pair[p]["total"] * 100 for p in pairs]

    colors = ['#3498db' if rate > 50 else '#e67e22' for rate in treat_rates]
    axes[1, 1].barh(pairs, treat_rates, color=colors)
    axes[1, 1].set_title("Treatment Rate by Agent-Principal Pair", fontweight='bold')
    axes[1, 1].set_xlabel("Treatment Rate (%)")
    axes[1, 1].grid(axis='x', alpha=0.3)
    axes[1, 1].axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    axes[1, 1].legend()

    plt.tight_layout()
    output_file = output_dir / "decision_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_belief_distribution(
    results: list[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    """Plot belief distribution across different groupings."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Belief Distribution Analysis", fontsize=16, fontweight='bold')

    # Extract beliefs
    all_beliefs = [float(r["belief"]) for r in results if r["belief"]]

    # 1. Overall belief histogram
    axes[0, 0].hist(all_beliefs, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(all_beliefs), color='red', linestyle='--',
                       label=f'Mean: {np.mean(all_beliefs):.3f}')
    axes[0, 0].axvline(np.median(all_beliefs), color='green', linestyle='--',
                       label=f'Median: {np.median(all_beliefs):.3f}')
    axes[0, 0].set_title("Overall Belief Distribution", fontweight='bold')
    axes[0, 0].set_xlabel("Belief Value")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Belief distribution by agent type
    by_agent = defaultdict(list)
    for r in results:
        if r["belief"]:
            by_agent[r["agent_name"]].append(float(r["belief"]))

    agent_beliefs = [by_agent[a] for a in by_agent.keys()]
    axes[0, 1].boxplot(agent_beliefs, tick_labels=list(by_agent.keys()), patch_artist=True,
                       boxprops=dict(facecolor='#3498db', alpha=0.7))
    axes[0, 1].set_title("Belief Distribution by Agent Type", fontweight='bold')
    axes[0, 1].set_ylabel("Belief Value")
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=15)

    # 3. Belief distribution by principal type
    by_principal = defaultdict(list)
    for r in results:
        if r["belief"]:
            by_principal[r["principal_name"]].append(float(r["belief"]))

    principal_beliefs = [by_principal[p] for p in by_principal.keys()]
    axes[1, 0].boxplot(principal_beliefs, tick_labels=list(by_principal.keys()), patch_artist=True,
                       boxprops=dict(facecolor='#e74c3c', alpha=0.7))
    axes[1, 0].set_title("Belief Distribution by Principal Type", fontweight='bold')
    axes[1, 0].set_ylabel("Belief Value")
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=15)

    # 4. Belief vs Decision scatter
    treat_beliefs = [float(r["belief"]) for r in results if r["decision"] == "treat" and r["belief"]]
    no_treat_beliefs = [float(r["belief"]) for r in results if r["decision"] == "do not treat" and r["belief"]]

    axes[1, 1].violinplot([treat_beliefs, no_treat_beliefs], positions=[1, 2],
                          showmeans=True, showmedians=True)
    axes[1, 1].set_title("Belief Distribution by Decision", fontweight='bold')
    axes[1, 1].set_ylabel("Belief Value")
    axes[1, 1].set_xticks([1, 2])
    axes[1, 1].set_xticklabels(['Treat', 'Do Not Treat'])
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "belief_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_heatmap_analysis(
    results: list[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    """Create heatmap visualizations of agent-principal interactions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Agent-Principal Interaction Heatmaps", fontsize=16, fontweight='bold')

    # Get unique agent and principal names
    agent_names = sorted(set(r["agent_name"] for r in results))
    principal_names = sorted(set(r["principal_name"] for r in results))

    # 1. Treatment rate heatmap
    treat_rate_matrix = np.zeros((len(agent_names), len(principal_names)))
    for i, agent in enumerate(agent_names):
        for j, principal in enumerate(principal_names):
            pair_results = [r for r in results
                          if r["agent_name"] == agent and r["principal_name"] == principal]
            if pair_results:
                treat_count = sum(1 for r in pair_results if r["decision"] == "treat")
                treat_rate_matrix[i, j] = (treat_count / len(pair_results)) * 100

    sns.heatmap(treat_rate_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=principal_names, yticklabels=agent_names,
                cbar_kws={'label': 'Treatment Rate (%)'}, ax=axes[0],
                vmin=0, vmax=100)
    axes[0].set_title("Treatment Rate (%)", fontweight='bold')
    axes[0].set_xlabel("Principal Type")
    axes[0].set_ylabel("Agent Type")

    # 2. Average belief heatmap
    belief_matrix = np.zeros((len(agent_names), len(principal_names)))
    for i, agent in enumerate(agent_names):
        for j, principal in enumerate(principal_names):
            pair_results = [r for r in results
                          if r["agent_name"] == agent and r["principal_name"] == principal]
            if pair_results:
                beliefs = [float(r["belief"]) for r in pair_results if r["belief"]]
                belief_matrix[i, j] = np.mean(beliefs) if beliefs else 0

    sns.heatmap(belief_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=principal_names, yticklabels=agent_names,
                cbar_kws={'label': 'Average Belief'}, ax=axes[1],
                vmin=0, vmax=1)
    axes[1].set_title("Average Belief", fontweight='bold')
    axes[1].set_xlabel("Principal Type")
    axes[1].set_ylabel("Agent Type")

    plt.tight_layout()
    output_file = output_dir / "heatmap_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_persuasion_effect(
    results: list[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    """Analyze persuasion effect: how agent type influences principal decisions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Agent Persuasion Effect Analysis", fontsize=16, fontweight='bold')

    # Get unique types
    agent_names = sorted(set(r["agent_name"] for r in results))
    principal_names = sorted(set(r["principal_name"] for r in results))

    # 1. Treatment rate by principal for each agent
    width = 0.35
    x = np.arange(len(principal_names))

    for i, agent in enumerate(agent_names):
        treat_rates = []
        for principal in principal_names:
            pair_results = [r for r in results
                          if r["agent_name"] == agent and r["principal_name"] == principal]
            if pair_results:
                treat_count = sum(1 for r in pair_results if r["decision"] == "treat")
                treat_rates.append((treat_count / len(pair_results)) * 100)
            else:
                treat_rates.append(0)

        offset = width * (i - len(agent_names)/2 + 0.5)
        axes[0].bar(x + offset, treat_rates, width, label=agent, alpha=0.8)

    axes[0].set_title("Treatment Rate by Principal for Each Agent", fontweight='bold')
    axes[0].set_xlabel("Principal Type")
    axes[0].set_ylabel("Treatment Rate (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(principal_names, rotation=15, ha='right')
    axes[0].legend(title="Agent Type")
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=50, color='red', linestyle='--', alpha=0.5)

    # 2. Average belief by principal for each agent
    for i, agent in enumerate(agent_names):
        avg_beliefs = []
        for principal in principal_names:
            pair_results = [r for r in results
                          if r["agent_name"] == agent and r["principal_name"] == principal]
            if pair_results:
                beliefs = [float(r["belief"]) for r in pair_results if r["belief"]]
                avg_beliefs.append(np.mean(beliefs) if beliefs else 0)
            else:
                avg_beliefs.append(0)

        offset = width * (i - len(agent_names)/2 + 0.5)
        axes[1].bar(x + offset, avg_beliefs, width, label=agent, alpha=0.8)

    axes[1].set_title("Average Belief by Principal for Each Agent", fontweight='bold')
    axes[1].set_xlabel("Principal Type")
    axes[1].set_ylabel("Average Belief")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(principal_names, rotation=15, ha='right')
    axes[1].legend(title="Agent Type")
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_file = output_dir / "persuasion_effect.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def save_statistics(stats: Mapping[str, Any], output_file: Path) -> None:
    """Save statistics to JSON file."""
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze principal-agent simulation results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to folder containing input results JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/output/analysis",
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and combine results from all files in input folder
    results = load_results_from_folder(args.input)
    print(f"Loaded {len(results)} total experiment results")

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(results)

    # Save statistics
    stats_file = output_dir / "statistics.json"
    save_statistics(stats, stats_file)

    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nOverall Statistics:")
    print(f"  Total decisions: {stats['overall_stats']['total_decisions']}")
    print(f"  Treatment rate: {stats['overall_stats']['treat_rate']:.2%}")

    print(f"\nBy Agent Type:")
    for agent, agent_stats in stats['by_agent'].items():
        print(f"  {agent}:")
        print(f"    Treatment rate: {agent_stats['treat_rate']:.2%}")
        print(f"    Avg belief: {agent_stats['avg_belief']:.3f}")

    print(f"\nBy Principal Type:")
    for principal, principal_stats in stats['by_principal'].items():
        print(f"  {principal}:")
        print(f"    Treatment rate: {principal_stats['treat_rate']:.2%}")
        print(f"    Avg belief: {principal_stats['avg_belief']:.3f}")

    print(f"\nBy Agent-Principal Pairs:")
    for pair, pair_stats in stats['by_pair'].items():
        print(f"  {pair_stats['agent_name']} + {pair_stats['principal_name']}:")
        print(f"    Treatment rate: {pair_stats['treat_rate']:.2%}")
        print(f"    Avg belief: {pair_stats['avg_belief']:.3f} (std: {pair_stats['std_belief']:.3f})")

    # Generate plots
    if not args.no_plots:
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        print("\nGenerating decision distribution plots...")
        plot_decision_distribution(results, output_dir)

        print("Generating belief distribution plots...")
        plot_belief_distribution(results, output_dir)

        print("Generating heatmap analysis...")
        plot_heatmap_analysis(results, output_dir)

        print("Generating persuasion effect analysis...")
        plot_persuasion_effect(results, output_dir)

        print(f"\n All visualizations saved to: {output_dir}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
