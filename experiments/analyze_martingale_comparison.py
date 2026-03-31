#!/usr/bin/env python3
"""Compare Bayesian martingale reliability across all agents and conditions.

The martingale property of Bayesian inference requires that a true Bayesian
should reach the same posterior regardless of the order in which evidence is
presented.  This script tests that property by comparing principal decisions
across k=3 random permutations of the same claim set.

Metrics
-------
  single_acc         Accuracy on permutation 0 (single-run baseline)
  majority_acc       Accuracy under majority vote across k permutations
  vote_gain          majority_acc - single_acc  (benefit of repeated sampling)
  mean_belief_std    Mean std of confidence across permutations (0 = perfect martingale)
  consistency        Mean fraction of permutations agreeing with majority answer
  perf_consist       Fraction of cases where ALL permutations agree (perfect invariance)
  violation_rate     1 - consistency  (fraction of decisions disrupted by reordering)

Cross-condition comparison
--------------------------
For baseline, framing, and information the script loads the *_choices.json
principal files (single-run, no permutations) and reports accuracy alongside
the martingale metrics so they can be read in the same table.

Usage
-----
    python experiments/analyze_martingale_comparison.py
    python experiments/analyze_martingale_comparison.py --output experiments/analysis/martingale_comparison.json
    python experiments/analyze_martingale_comparison.py --detail deepseek
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

_SCRIPT_DIR    = Path(__file__).parent
PRINCIPALS_DIR = _SCRIPT_DIR / "principals/usmle_sample"
DEFAULT_OUTPUT = str(_SCRIPT_DIR / "analysis/martingale_comparison.json")

AGENTS       = ["claude", "deepseek", "gemini", "gpt", "llama", "llama-dpo", "llama-sft", "llama-small"]
PROMPT_TYPES = ["bayesian", "behavioral"]

_CHOICES_PATTERNS = {
    "baseline":    "principal_{a}_{pt}_choices.json",
    "framing":     "principal_framing_{a}_gt_factual_agg_{pt}_choices.json",
    "information": "principal_information_{a}_gt_factual_agg_{pt}_choices.json",
}
_MARTINGALE_PRINCIPAL = "principal_martingale_{a}_k3_bayesian_martingale_choices.json"
_MARTINGALE_ANALYSIS  = "martingale_analysis_{a}_k3.json"
_BASE_PRINCIPAL       = "principal_martingale_k3_bayesian_martingale_choices.json"
_BASE_ANALYSIS        = "martingale_analysis_k3.json"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_choices_accuracy(agent: str, condition: str, pt: str) -> dict | None:
    path = PRINCIPALS_DIR / _CHOICES_PATTERNS[condition].format(a=agent, pt=pt)
    entries = load_json(path)
    if not entries:
        return None
    total   = len(entries)
    correct = sum(
        1 for e in entries
        if (e.get("decision") or "").strip().upper() ==
           (e.get("correct_answer_idx") or "").strip().upper()
    )
    return {"correct": correct, "total": total, "accuracy": correct / total}


# ── Martingale computation (from raw principal file) ─────────────────────────

def compute_martingale_stats(entries: list[dict]) -> dict:
    """Recompute martingale metrics from a flat list of per-permutation entries."""
    by_case: dict[str, list[dict]] = {}
    for e in entries:
        by_case.setdefault(e["case_id"], []).append(e)

    per_case = []
    for cid, perms in sorted(by_case.items()):
        answers = [(e.get("decision") or "").strip().upper() for e in perms if e.get("decision")]
        beliefs = []
        for e in perms:
            try:
                b = float(str(e.get("belief", "")).strip())
                if 0.0 <= b <= 1.0:
                    beliefs.append(b)
            except (ValueError, TypeError):
                pass

        gt = (perms[0].get("correct_answer_idx") or "").strip().upper()
        if not answers or not gt:
            continue

        counter        = Counter(answers)
        majority, cnt  = counter.most_common(1)[0]
        consistency    = cnt / len(answers)
        mean_bel       = statistics.mean(beliefs) if beliefs else None
        std_bel        = statistics.stdev(beliefs) if len(beliefs) >= 2 else 0.0

        per_case.append({
            "case_id":          cid,
            "majority_answer":  majority,
            "majority_correct": majority == gt,
            "answer_consistency": consistency,
            "mean_belief":      round(mean_bel, 4) if mean_bel is not None else None,
            "std_belief":       round(std_bel, 4),
            "correct_answer_idx": gt,
            "num_permutations": len(perms),
        })

    if not per_case:
        return {}

    n              = len(per_case)
    majority_acc   = sum(1 for c in per_case if c["majority_correct"]) / n
    stds           = [c["std_belief"] for c in per_case]
    consistencies  = [c["answer_consistency"] for c in per_case]
    perf_consist   = sum(1 for c in consistencies if c == 1.0) / n

    # Single-run accuracy (permutation_idx == 0)
    first_map = {e["case_id"]: e for e in entries if e.get("permutation_idx", 0) == 0}
    single_correct = sum(
        1 for e in first_map.values()
        if (e.get("decision") or "").strip().upper() ==
           (e.get("correct_answer_idx") or "").strip().upper()
    )
    single_acc = single_correct / len(first_map) if first_map else None

    return {
        "n_cases":                   n,
        "single_acc":                round(single_acc, 4) if single_acc is not None else None,
        "majority_acc":              round(majority_acc, 4),
        "vote_gain":                 round(majority_acc - single_acc, 4) if single_acc is not None else None,
        "mean_belief_std":           round(statistics.mean(stds), 4),
        "median_belief_std":         round(statistics.median(stds), 4),
        "mean_answer_consistency":   round(statistics.mean(consistencies), 4),
        "fraction_perfect_consistency": round(perf_consist, 4),
        "violation_rate":            round(1 - statistics.mean(consistencies), 4),
        "per_case":                  per_case,
    }


# ── Build full results ────────────────────────────────────────────────────────

def build_results() -> dict:
    results: dict = {}

    # Base condition (no agent — raw claim permutations)
    base_entries = load_json(PRINCIPALS_DIR / _BASE_PRINCIPAL)
    results["(base)"] = {
        "martingale": compute_martingale_stats(base_entries) if base_entries else None,
        "conditions":  {},
    }

    for agent in AGENTS:
        # Martingale stats
        mart_entries = load_json(PRINCIPALS_DIR / _MARTINGALE_PRINCIPAL.format(a=agent))
        mart_stats   = compute_martingale_stats(mart_entries) if mart_entries else None

        # Single-run accuracies per condition x prompt type
        conditions: dict = {}
        for cond in ["baseline", "framing", "information"]:
            conditions[cond] = {}
            for pt in PROMPT_TYPES:
                conditions[cond][pt] = load_choices_accuracy(agent, cond, pt)

        results[agent] = {
            "martingale": mart_stats,
            "conditions": conditions,
        }

    return results


# ── Printing ──────────────────────────────────────────────────────────────────

def print_martingale_table(results: dict) -> None:
    print("=== TABLE 1: Martingale Reliability (k=3 permutations, Bayesian principal) ===\n")
    hdr = (f"{'Agent':<12}  {'n_cases':>7}  {'single_acc':>10}  {'majority_acc':>12}"
           f"  {'vote_gain':>9}  {'bel_std':>7}  {'consistency':>11}  {'perf_consist':>12}  {'violation':>9}")
    print(hdr)
    print("-" * len(hdr))

    for agent in ["(base)"] + AGENTS:
        m = results[agent].get("martingale")
        if not m:
            print(f"{agent:<12}  N/A")
            continue
        vote_gain = f"{m['vote_gain']:+.3f}" if m["vote_gain"] is not None else "  N/A"
        single    = f"{m['single_acc']:.3f}" if m["single_acc"] is not None else " N/A"
        print(
            f"{agent:<12}  {m['n_cases']:>7}  {single:>10}  {m['majority_acc']:>12.3f}"
            f"  {vote_gain:>9}  {m['mean_belief_std']:>7.4f}"
            f"  {m['mean_answer_consistency']:>11.4f}  {m['fraction_perfect_consistency']:>12.4f}"
            f"  {m['violation_rate']:>9.4f}"
        )


def print_cross_condition_table(results: dict) -> None:
    print("\n=== TABLE 2: Accuracy Across All Conditions (Bayesian principal) ===\n")
    hdr = (f"{'Agent':<12}  {'base':>6}  {'framing':>7}  {'info':>6}"
           f"  {'mart_single':>11}  {'mart_majority':>13}  {'vote_gain':>9}")
    print(hdr)
    print("-" * len(hdr))

    for agent in AGENTS:
        r = results[agent]
        m = r.get("martingale")
        c = r.get("conditions", {})

        def acc(cond, pt="bayesian"):
            v = c.get(cond, {}).get(pt)
            return f"{v['accuracy']:.3f}" if v else "  N/A"

        mart_single = f"{m['single_acc']:.3f}" if m and m["single_acc"] is not None else "  N/A"
        mart_maj    = f"{m['majority_acc']:.3f}" if m else "  N/A"
        vote_gain   = f"{m['vote_gain']:+.3f}" if m and m["vote_gain"] is not None else "  N/A"

        print(f"{agent:<12}  {acc('baseline'):>6}  {acc('framing'):>7}  {acc('information'):>6}"
              f"  {mart_single:>11}  {mart_maj:>13}  {vote_gain:>9}")

    print("\n--- Behavioral principal ---\n")
    print(hdr)
    print("-" * len(hdr))
    for agent in AGENTS:
        r = results[agent]
        m = r.get("martingale")
        c = r.get("conditions", {})

        def acc(cond, pt="behavioral"):
            v = c.get(cond, {}).get(pt)
            return f"{v['accuracy']:.3f}" if v else "  N/A"

        # Martingale is bayesian-only
        mart_single = f"{m['single_acc']:.3f}" if m and m["single_acc"] is not None else "  N/A"
        mart_maj    = f"{m['majority_acc']:.3f}" if m else "  N/A"
        vote_gain   = f"{m['vote_gain']:+.3f}" if m and m["vote_gain"] is not None else "  N/A"

        print(f"{agent:<12}  {acc('baseline'):>6}  {acc('framing'):>7}  {acc('information'):>6}"
              f"  {mart_single:>11}  {mart_maj:>13}  {vote_gain:>9}")


def print_violation_vs_harm(results: dict) -> None:
    print("\n=== TABLE 3: Martingale Violations vs Decision Harms ===\n")
    print("Cases where reordering changes the answer (violation) vs. harm from framing/info.\n")
    hdr = (f"{'Agent':<12}  {'violation_rate':>14}  {'perf_consist':>12}"
           f"  {'inconsist_cases':>15}  {'bel_std':>7}")
    print(hdr)
    print("-" * len(hdr))
    for agent in AGENTS:
        m = results[agent].get("martingale")
        if not m:
            print(f"{agent:<12}  N/A")
            continue
        n         = m["n_cases"]
        viol_n    = round(m["violation_rate"] * n)
        print(
            f"{agent:<12}  {m['violation_rate']:>14.4f}  {m['fraction_perfect_consistency']:>12.4f}"
            f"  {viol_n:>15}  {m['mean_belief_std']:>7.4f}"
        )


def print_detail(results: dict, agent: str) -> None:
    m = results.get(agent, {}).get("martingale")
    if not m or not m.get("per_case"):
        print(f"No martingale data for {agent}")
        return
    per_case = m["per_case"]
    inconsistent = [c for c in per_case if c["answer_consistency"] < 1.0]
    print(f"\n=== DETAIL: {agent} — {len(inconsistent)} inconsistent cases "
          f"({len(inconsistent)/len(per_case):.1%} of {len(per_case)}) ===\n")
    print(f"{'case_id':<22}  {'majority':>8}  {'correct':>7}  {'consistent':>10}  {'bel_std':>7}  {'gt':>4}")
    print("-" * 70)
    for c in sorted(inconsistent, key=lambda x: x["answer_consistency"]):
        correct_mark = "✓" if c["majority_correct"] else "✗"
        print(f"{c['case_id']:<22}  {c['majority_answer']:>8}  {correct_mark:>7}"
              f"  {c['answer_consistency']:>10.3f}  {c['std_belief']:>7.4f}  {c['correct_answer_idx']:>4}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output",  default=DEFAULT_OUTPUT)
    parser.add_argument("--detail",  metavar="AGENT", help="Print per-case inconsistencies for a specific agent")
    args = parser.parse_args()

    results = build_results()

    print_martingale_table(results)
    print_cross_condition_table(results)
    print_violation_vs_harm(results)

    if args.detail:
        print_detail(results, args.detail)

    # Export: strip per_case from martingale to keep JSON compact
    export: dict = {}
    for agent, data in results.items():
        export[agent] = {
            "martingale": {k: v for k, v in (data["martingale"] or {}).items() if k != "per_case"},
            "conditions": data["conditions"],
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(export, indent=2))
    print(f"\nSaved results → {out_path}")


if __name__ == "__main__":
    main()
