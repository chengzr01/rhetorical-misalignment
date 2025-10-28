#!/usr/bin/env python3
"""
Compare decisions made by principals in DPO vs SFT settings.

This script compares the treatment decisions made by principals across
two different training settings (e.g., DPO vs SFT) and identifies cases
where the decisions differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def extract_patient_id(context: str) -> str:
    """Extract patient ID from principal context."""
    lines = context.split('\n')
    for line in lines:
        if line.startswith('Patient ID:'):
            return line.split(':', 1)[1].strip()
    return 'Unknown'


def load_results(file_path: Path) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def compare_decisions(
    dpo_results: List[Dict[str, Any]],
    sft_results: List[Dict[str, Any]],
    show_all: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Compare decisions between DPO and SFT results.

    Args:
        dpo_results: List of results from DPO setting
        sft_results: List of results from SFT setting
        show_all: If True, show all cases; if False, only show differences

    Returns:
        Tuple of (list of comparison results, summary statistics)
    """
    if len(dpo_results) != len(sft_results):
        print(f"Warning: Different number of results (DPO: {len(dpo_results)}, SFT: {len(sft_results)})")
        print("Comparing only the common length.")

    comparisons = []
    stats = {
        'total': 0,
        'different': 0,
        'dpo_treat': 0,
        'dpo_no_treat': 0,
        'sft_treat': 0,
        'sft_no_treat': 0,
        'both_treat': 0,
        'both_no_treat': 0,
    }

    min_len = min(len(dpo_results), len(sft_results))

    for i in range(min_len):
        dpo = dpo_results[i]
        sft = sft_results[i]

        patient_id = extract_patient_id(dpo.get('principal_context', ''))

        dpo_decision = dpo.get('decision', 'unknown')
        sft_decision = sft.get('decision', 'unknown')

        decisions_differ = dpo_decision != sft_decision

        # Update statistics
        stats['total'] += 1
        if decisions_differ:
            stats['different'] += 1

        if dpo_decision == 'treat':
            stats['dpo_treat'] += 1
        else:
            stats['dpo_no_treat'] += 1

        if sft_decision == 'treat':
            stats['sft_treat'] += 1
        else:
            stats['sft_no_treat'] += 1

        if dpo_decision == 'treat' and sft_decision == 'treat':
            stats['both_treat'] += 1
        elif dpo_decision == 'do not treat' and sft_decision == 'do not treat':
            stats['both_no_treat'] += 1

        if show_all or decisions_differ:
            comparison = {
                'index': i,
                'patient_id': patient_id,
                'dpo_decision': dpo_decision,
                'sft_decision': sft_decision,
                'decisions_differ': decisions_differ,
                'dpo_belief': dpo.get('belief', 'N/A'),
                'sft_belief': sft.get('belief', 'N/A'),
                'principal_context': dpo.get('principal_context', ''),
                'agent_information_dpo': dpo.get('information', ''),
                'agent_information_sft': sft.get('information', ''),
                'dpo_reasoning': dpo.get('reasoning', ''),
                'sft_reasoning': sft.get('reasoning', ''),
            }
            comparisons.append(comparison)

    return comparisons, stats


def print_summary(stats: Dict[str, int]) -> None:
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total cases compared: {stats['total']}")
    print(f"Cases with different decisions: {stats['different']} ({stats['different']/stats['total']*100:.1f}%)")
    print(f"Cases with same decisions: {stats['total'] - stats['different']} ({(stats['total'] - stats['different'])/stats['total']*100:.1f}%)")
    print()
    print(f"DPO decisions:")
    print(f"  - Treat: {stats['dpo_treat']} ({stats['dpo_treat']/stats['total']*100:.1f}%)")
    print(f"  - Do not treat: {stats['dpo_no_treat']} ({stats['dpo_no_treat']/stats['total']*100:.1f}%)")
    print()
    print(f"SFT decisions:")
    print(f"  - Treat: {stats['sft_treat']} ({stats['sft_treat']/stats['total']*100:.1f}%)")
    print(f"  - Do not treat: {stats['sft_no_treat']} ({stats['sft_no_treat']/stats['total']*100:.1f}%)")
    print()
    print(f"Agreement:")
    print(f"  - Both treat: {stats['both_treat']} ({stats['both_treat']/stats['total']*100:.1f}%)")
    print(f"  - Both do not treat: {stats['both_no_treat']} ({stats['both_no_treat']/stats['total']*100:.1f}%)")
    print("="*80)


def print_comparison(comparison: Dict[str, Any], verbose: bool = False) -> None:
    """Print a single comparison case."""
    print("\n" + "-"*80)
    print(f"Case #{comparison['index']} | Patient ID: {comparison['patient_id']}")
    print("-"*80)

    # Extract hypothesis from context
    context_lines = comparison['principal_context'].split('\n\n')
    hypothesis = "Not found"
    for line in context_lines:
        if line.startswith('Hypothesis:'):
            hypothesis = line.replace('Hypothesis:', '').strip()
            break

    print(f"\nHypothesis: {hypothesis[:200]}...")

    print(f"\nDECISIONS:")
    differ_marker = " *** DIFFERENT ***" if comparison['decisions_differ'] else " (same)"
    print(f"  DPO: {comparison['dpo_decision'].upper()} (belief: {comparison['dpo_belief']}){differ_marker}")
    print(f"  SFT: {comparison['sft_decision'].upper()} (belief: {comparison['sft_belief']}){differ_marker}")

    if verbose:
        print(f"\n--- DPO REASONING ---")
        print(comparison['dpo_reasoning'][:500] + "..." if len(comparison['dpo_reasoning']) > 500 else comparison['dpo_reasoning'])

        print(f"\n--- SFT REASONING ---")
        print(comparison['sft_reasoning'][:500] + "..." if len(comparison['sft_reasoning']) > 500 else comparison['sft_reasoning'])

        print(f"\n--- DPO AGENT INFORMATION ---")
        print(comparison['agent_information_dpo'][:500] + "..." if len(comparison['agent_information_dpo']) > 500 else comparison['agent_information_dpo'])

        print(f"\n--- SFT AGENT INFORMATION ---")
        print(comparison['agent_information_sft'][:500] + "..." if len(comparison['agent_information_sft']) > 500 else comparison['agent_information_sft'])


def save_differences(comparisons: List[Dict[str, Any]], output_path: Path) -> None:
    """Save comparison results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(comparisons, f, indent=2)
    print(f"\nComparison results saved to: {output_path}")


def save_summary(stats: Dict[str, int], output_path: Path) -> None:
    """Save summary statistics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSummary statistics saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare decisions between DPO and SFT settings"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['mimic', 'pmc'],
        default='mimic',
        help="Dataset to compare (mimic or pmc)"
    )
    parser.add_argument(
        "--dpo",
        type=str,
        help="Path to DPO results JSON file"
    )
    parser.add_argument(
        "--sft",
        type=str,
        help="Path to SFT results JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path to save comparison results (optional)"
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all cases, not just differences"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed reasoning and agent information"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of differences to display"
    )

    args = parser.parse_args()

    # Set default paths based on dataset if not provided
    if not args.dpo:
        args.dpo = f"experiments/output/{args.dataset}_results_dpo.json"
    if not args.sft:
        args.sft = f"experiments/output/{args.dataset}_results_sft.json"

    dpo_path = Path(args.dpo)
    sft_path = Path(args.sft)

    if not dpo_path.exists():
        print(f"Error: DPO file not found: {dpo_path}")
        return

    if not sft_path.exists():
        print(f"Error: SFT file not found: {sft_path}")
        return

    print(f"Loading DPO results from: {dpo_path}")
    dpo_results = load_results(dpo_path)
    print(f"Loaded {len(dpo_results)} DPO results")

    print(f"Loading SFT results from: {sft_path}")
    sft_results = load_results(sft_path)
    print(f"Loaded {len(sft_results)} SFT results")

    comparisons, stats = compare_decisions(dpo_results, sft_results, args.show_all)

    print_summary(stats)

    # Save summary statistics
    if args.output:
        output_dir = Path(args.output).parent
        summary_path = output_dir / f"{args.dataset}_summary.json"
        save_summary(stats, summary_path)

    if comparisons:
        display_limit = args.limit if args.limit else len(comparisons)
        print(f"\n{'='*80}")
        print(f"{'DISPLAYING ' + str(min(display_limit, len(comparisons))) + ' CASES'}")
        if not args.show_all:
            print("(Showing only cases where decisions differ)")
        print(f"{'='*80}")

        for i, comparison in enumerate(comparisons[:display_limit]):
            print_comparison(comparison, verbose=args.verbose)
    else:
        if not args.show_all:
            print("\nNo differences found! All decisions match.")

    if args.output:
        output_path = Path(args.output)
        save_differences(comparisons, output_path)


if __name__ == "__main__":
    main()
