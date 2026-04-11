#!/usr/bin/env python3
"""Analyze martingale reliability of the Bayesian benchmark.

Computes per-case and aggregate martingale statistics across all agents and conditions,
covering four sub-questions:

  Q1. Does permutation sampling improve accuracy?
      (majority-vote accuracy vs. single-run, fragile-case recovery)
  Q2. How does framing affect martingale compliance?
      (violation rate, belief std, fragile rate per agent)
  Q3. How does information design relate to the martingale property?
      (claim-selection stats as a proxy for ordering sensitivity)
  Q4. Do martingale-fragile cases overlap with framing/info harm cases?

Also prints a cross-condition accuracy table (baseline, framing, info, martingale).

Usage:
    python experiments/analyze_martingale.py
    python experiments/analyze_martingale.py --output experiments/analysis/martingale.json
    python experiments/analyze_martingale.py --detail deepseek
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

_SCRIPT_DIR    = Path(__file__).parent
_PROJECT_DIR   = _SCRIPT_DIR.parent
PRINCIPALS_DIR = _PROJECT_DIR / "experiments/principals/usmle_sample"
AGENTS_DIR     = _PROJECT_DIR / "experiments/agents/usmle_sample"
DEFAULT_OUTPUT = str(_PROJECT_DIR / "experiments/analysis/martingale.json")

AGENTS       = ["claude", "deepseek", "gemini", "gpt", "llama", "llama-dpo", "llama-sft", "llama-small"]
PROMPT_TYPES = ["bayesian", "behavioral"]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> list | dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_martingale(agent: str | None) -> list[dict] | None:
    fname = ("principal_martingale_k3_bayesian_martingale_choices.json" if agent is None
             else f"principal_martingale_{agent}_k3_bayesian_martingale_choices.json")
    return load_json(PRINCIPALS_DIR / fname)


def load_choices(agent: str, condition: str, pt: str) -> list[dict] | None:
    patterns = {
        "baseline":    f"principal_{agent}_{pt}_choices.json",
        "framing":     f"principal_framing_{agent}_gt_factual_agg_{pt}_choices.json",
        "information": f"principal_information_{agent}_gt_factual_agg_{pt}_choices.json",
    }
    return load_json(PRINCIPALS_DIR / patterns[condition])


def load_choices_accuracy(agent: str, condition: str, pt: str) -> dict | None:
    entries = load_choices(agent, condition, pt)
    if not entries:
        return None
    total   = len(entries)
    correct = sum(
        1 for e in entries
        if (e.get("decision") or "").strip().upper() ==
           (e.get("correct_answer_idx") or "").strip().upper()
    )
    return {"correct": correct, "total": total, "accuracy": correct / total}


def load_choices_map(agent: str, condition: str, pt: str = "bayesian") -> dict[str, dict] | None:
    entries = load_choices(agent, condition, pt)
    return {e["case_id"]: e for e in entries} if entries else None


def load_agent_selection(agent: str) -> dict[str, dict] | None:
    entries = load_json(AGENTS_DIR / f"information_{agent}_gt_factual_agg.json")
    return {e["case_id"]: e for e in entries} if entries else None


# ── Per-case computation ──────────────────────────────────────────────────────

def per_case_stats(entries: list[dict]) -> list[dict]:
    by_case: dict[str, list[dict]] = {}
    for e in entries:
        by_case.setdefault(e["case_id"], []).append(e)

    records = []
    for cid, perms in sorted(by_case.items()):
        answers = [(e.get("decision") or "").strip().upper() for e in perms if e.get("decision")]
        gt      = (perms[0].get("correct_answer_idx") or "").strip().upper()
        if not answers or not gt:
            continue

        beliefs = []
        for e in perms:
            try:
                b = float(str(e.get("belief", "")).strip())
                if 0.0 <= b <= 1.0:
                    beliefs.append(b)
            except (ValueError, TypeError):
                pass

        counter       = Counter(answers)
        majority, cnt = counter.most_common(1)[0]
        consistency   = cnt / len(answers)
        perm0         = answers[0]
        std_belief    = statistics.stdev(beliefs) if len(beliefs) >= 2 else 0.0
        mean_belief   = statistics.mean(beliefs) if beliefs else None

        records.append({
            "case_id":          cid,
            "gt":               gt,
            "perm0_answer":     perm0,
            "majority_answer":  majority,
            "perm0_correct":    perm0 == gt,
            "majority_correct": majority == gt,
            "consistency":      consistency,
            "fragile":          consistency < 1.0,
            "mean_belief":      round(mean_belief, 4) if mean_belief is not None else None,
            "std_belief":       round(std_belief, 4),
            "num_permutations": len(perms),
            "meta_info":        perms[0].get("meta_info"),
        })
    return records


def summarise(records: list[dict]) -> dict:
    if not records:
        return {}
    n            = len(records)
    single_acc   = sum(1 for r in records if r["perm0_correct"]) / n
    majority_acc = sum(1 for r in records if r["majority_correct"]) / n
    vote_helps   = sum(1 for r in records if not r["perm0_correct"] and r["majority_correct"])
    vote_hurts   = sum(1 for r in records if r["perm0_correct"] and not r["majority_correct"])
    fragile      = [r for r in records if r["fragile"]]
    n_frag       = len(fragile)
    frag_correct = sum(1 for r in fragile if r["majority_correct"])
    stds         = [r["std_belief"] for r in records]
    frag_stds    = [r["std_belief"] for r in fragile]
    stable_stds  = [r["std_belief"] for r in records if not r["fragile"]]
    consistencies = [r["consistency"] for r in records]
    perf_consist  = sum(1 for c in consistencies if c == 1.0) / n

    frag_by_step: dict[str, int] = {}
    for r in fragile:
        step = r.get("meta_info") or "unknown"
        frag_by_step[step] = frag_by_step.get(step, 0) + 1

    return {
        "n_cases":                   n,
        "single_acc":                round(single_acc, 4),
        "majority_acc":              round(majority_acc, 4),
        "vote_gain":                 round(majority_acc - single_acc, 4),
        "vote_helps":                vote_helps,
        "vote_hurts":                vote_hurts,
        "n_fragile":                 n_frag,
        "fragile_rate":              round(n_frag / n, 4),
        "frag_majority_correct":     frag_correct,
        "frag_recovery_rate":        round(frag_correct / n_frag, 4) if n_frag else None,
        "mean_belief_std":           round(statistics.mean(stds), 4),
        "mean_belief_std_fragile":   round(statistics.mean(frag_stds), 4) if frag_stds else None,
        "mean_belief_std_stable":    round(statistics.mean(stable_stds), 4) if stable_stds else None,
        "belief_std_ratio":          round(statistics.mean(frag_stds) / statistics.mean(stable_stds), 2)
                                     if frag_stds and stable_stds and statistics.mean(stable_stds) > 0 else None,
        "mean_consistency":          round(statistics.mean(consistencies), 4),
        "perf_consistency":          round(perf_consist, 4),
        "violation_rate":            round(1 - statistics.mean(consistencies), 4),
        "fragile_by_step":           frag_by_step,
    }


# ── Harm and selection helpers ────────────────────────────────────────────────

def get_harm_ids(baseline: dict | None, experiment: dict | None) -> set[str]:
    if not baseline or not experiment:
        return set()
    return {
        cid for cid, be in baseline.items()
        if (be.get("decision") or "").strip().upper() ==
           (be.get("correct_answer_idx") or "").strip().upper()
        and (experiment.get(cid, {}).get("decision") or "").strip().upper() !=
            (be.get("correct_answer_idx") or "").strip().upper()
    }


def selection_summary(agent: str) -> dict | None:
    sel = load_agent_selection(agent)
    if not sel:
        return None
    n_avail = [e["n_available"] for e in sel.values() if e.get("n_available")]
    n_sel   = [e["n_selected"]  for e in sel.values() if e.get("n_selected") is not None]
    if not n_avail:
        return None
    drops = [a - s for a, s in zip(n_avail, n_sel)]
    return {
        "n_cases":       len(n_avail),
        "avg_available": round(statistics.mean(n_avail), 1),
        "avg_selected":  round(statistics.mean(n_sel), 1),
        "avg_dropped":   round(statistics.mean(drops), 1),
        "avg_drop_rate": round(statistics.mean(d / a for d, a in zip(drops, n_avail)), 4),
    }


# ── Build full results ────────────────────────────────────────────────────────

def build_results() -> dict:
    results: dict = {}

    base_entries = load_martingale(None)
    results["(base)"] = {
        "martingale":    summarise(per_case_stats(base_entries)) if base_entries else None,
        "conditions":    {},
        "overlap":       {},
        "info_selection": None,
    }

    for agent in AGENTS:
        mart_entries = load_martingale(agent)
        mart_records = per_case_stats(mart_entries) if mart_entries else []

        base_map    = load_choices_map(agent, "baseline")
        framing_map = load_choices_map(agent, "framing")
        info_map    = load_choices_map(agent, "information")

        conditions = {
            cond: {pt: load_choices_accuracy(agent, cond, pt) for pt in PROMPT_TYPES}
            for cond in ["baseline", "framing", "information"]
        }

        fram_harms  = get_harm_ids(base_map, framing_map)
        info_harms  = get_harm_ids(base_map, info_map)
        fragile_ids = {r["case_id"] for r in mart_records if r["fragile"]}

        overlap: dict = {}
        if fragile_ids:
            overlap["fragile_and_framing_harm"] = len(fragile_ids & fram_harms)
            overlap["fragile_and_info_harm"]    = len(fragile_ids & info_harms)
            overlap["fragile_only"]             = len(fragile_ids - fram_harms - info_harms)

        results[agent] = {
            "martingale":    summarise(mart_records) if mart_records else None,
            "conditions":    conditions,
            "overlap":       overlap,
            "info_selection": selection_summary(agent),
        }

    return results


# ── Print tables ──────────────────────────────────────────────────────────────

def print_summary_table(results: dict) -> None:
    print("=== Martingale Reliability (k=3 permutations, Bayesian principal) ===\n")
    hdr = (f"{'Agent':<12}  {'n_cases':>7}  {'single_acc':>10}  {'majority_acc':>12}"
           f"  {'vote_gain':>9}  {'bel_std':>7}  {'consistency':>11}  {'perf_consist':>12}  {'violation':>9}")
    print(hdr)
    print("-" * len(hdr))
    for agent in ["(base)"] + AGENTS:
        m = results[agent].get("martingale")
        if not m:
            print(f"{agent:<12}  N/A"); continue
        vote_gain = f"{m['vote_gain']:+.3f}" if m.get("vote_gain") is not None else "  N/A"
        single    = f"{m['single_acc']:.3f}" if m.get("single_acc") is not None else " N/A"
        print(
            f"{agent:<12}  {m['n_cases']:>7}  {single:>10}  {m['majority_acc']:>12.3f}"
            f"  {vote_gain:>9}  {m['mean_belief_std']:>7.4f}"
            f"  {m['mean_consistency']:>11.4f}  {m['perf_consistency']:>12.4f}"
            f"  {m['violation_rate']:>9.4f}"
        )


def print_cross_condition_table(results: dict) -> None:
    print("\n=== Accuracy Across All Conditions ===\n")
    for pt in PROMPT_TYPES:
        print(f"--- {pt.capitalize()} principal ---\n")
        hdr = (f"{'Agent':<12}  {'baseline':>8}  {'framing':>7}  {'info':>6}"
               f"  {'mart_single':>11}  {'mart_majority':>13}  {'vote_gain':>9}")
        print(hdr)
        print("-" * len(hdr))
        for agent in AGENTS:
            r = results[agent]
            m = r.get("martingale")
            c = r.get("conditions", {})

            def acc(cond: str) -> str:
                v = c.get(cond, {}).get(pt)
                return f"{v['accuracy']:.3f}" if v else "  N/A"

            mart_single = f"{m['single_acc']:.3f}" if m and m.get("single_acc") is not None else "  N/A"
            mart_maj    = f"{m['majority_acc']:.3f}" if m else "  N/A"
            vote_gain   = f"{m['vote_gain']:+.3f}" if m and m.get("vote_gain") is not None else "  N/A"
            print(f"{agent:<12}  {acc('baseline'):>8}  {acc('framing'):>7}  {acc('information'):>6}"
                  f"  {mart_single:>11}  {mart_maj:>13}  {vote_gain:>9}")
        print()


def print_q1_permutation_benefit(results: dict) -> None:
    print("=== Q1: Does Permutation Improve Accuracy? ===\n")
    hdr = (f"{'Agent':<12}  {'single_acc':>10}  {'majority_acc':>12}  {'vote_gain':>9}"
           f"  {'vote_helps':>10}  {'vote_hurts':>10}  {'net':>4}")
    print(hdr)
    print("-" * len(hdr))
    for agent in ["(base)"] + AGENTS:
        m = results[agent].get("martingale")
        if not m:
            print(f"{agent:<12}  N/A"); continue
        net = m["vote_helps"] - m["vote_hurts"]
        print(f"{agent:<12}  {m['single_acc']:>10.3f}  {m['majority_acc']:>12.3f}"
              f"  {m['vote_gain']:>+9.3f}  {m['vote_helps']:>10}  {m['vote_hurts']:>10}  {net:>+4}")

    print("\n--- Fragile-case recovery ---\n")
    hdr2 = (f"{'Agent':<12}  {'n_fragile':>9}  {'frag_rate':>9}  {'frag_correct':>12}"
            f"  {'recovery':>8}  {'bel_std_stable':>14}  {'bel_std_fragile':>15}  {'ratio':>5}")
    print(hdr2)
    print("-" * len(hdr2))
    for agent in ["(base)"] + AGENTS:
        m = results[agent].get("martingale")
        if not m:
            print(f"{agent:<12}  N/A"); continue
        rec   = f"{m['frag_recovery_rate']:.3f}" if m.get("frag_recovery_rate") is not None else "  N/A"
        ratio = f"{m['belief_std_ratio']:.1f}x"  if m.get("belief_std_ratio")   is not None else "  N/A"
        s_std = f"{m['mean_belief_std_stable']:.4f}"  if m.get("mean_belief_std_stable")  is not None else "  N/A"
        f_std = f"{m['mean_belief_std_fragile']:.4f}" if m.get("mean_belief_std_fragile") is not None else "  N/A"
        print(f"{agent:<12}  {m['n_fragile']:>9}  {m['fragile_rate']:>9.3f}"
              f"  {m['frag_majority_correct']:>12}  {rec:>8}"
              f"  {s_std:>14}  {f_std:>15}  {ratio:>5}")


def print_q2_framing_effect(results: dict) -> None:
    print("\n=== Q2: How Does Framing Affect Martingale Compliance? ===\n")
    base_m = results["(base)"]["martingale"]
    if base_m:
        print(f"  Base:  violation={base_m['violation_rate']:.4f}  "
              f"fragile={base_m['fragile_rate']:.3f}  bel_std={base_m['mean_belief_std']:.4f}\n")
    hdr = (f"{'Agent':<12}  {'violation':>9}  {'vs_base':>7}  {'fragile_rate':>12}"
           f"  {'bel_std':>7}  {'perf_consist':>12}  {'single_acc':>10}  {'maj_acc':>7}")
    print(hdr)
    print("-" * len(hdr))
    for agent in AGENTS:
        m = results[agent].get("martingale")
        if not m:
            print(f"{agent:<12}  N/A"); continue
        viol_delta = m["violation_rate"] - (base_m["violation_rate"] if base_m else 0)
        print(f"{agent:<12}  {m['violation_rate']:>9.4f}  {viol_delta:>+7.4f}"
              f"  {m['fragile_rate']:>12.3f}  {m['mean_belief_std']:>7.4f}"
              f"  {m['perf_consistency']:>12.4f}  {m['single_acc']:>10.3f}  {m['majority_acc']:>7.3f}")

    print("\n--- Fragile cases by USMLE step ---\n")
    print(f"{'Agent':<12}  {'step1':>6}  {'step2':>6}  {'step3':>6}  {'total':>6}")
    print("-" * 44)
    for agent in AGENTS:
        m = results[agent].get("martingale")
        if not m:
            print(f"{agent:<12}  N/A"); continue
        s = m.get("fragile_by_step", {})
        print(f"{agent:<12}  {s.get('step1',0):>6}  {s.get('step2',0):>6}"
              f"  {s.get('step3',0):>6}  {m['n_fragile']:>6}")


def print_q3_info_design(results: dict) -> None:
    print("\n=== Q3: Information Design — Claim Selection Stats ===\n")
    print("  Fewer claims selected → lower expected ordering sensitivity.\n")
    hdr = (f"{'Agent':<12}  {'n_cases':>7}  {'avg_avail':>9}  {'avg_sel':>7}"
           f"  {'avg_drop':>8}  {'drop_rate':>9}")
    print(hdr)
    print("-" * 60)
    for agent in AGENTS:
        s = results[agent].get("info_selection")
        if not s:
            print(f"{agent:<12}  N/A"); continue
        print(f"{agent:<12}  {s['n_cases']:>7}  {s['avg_available']:>9.1f}"
              f"  {s['avg_selected']:>7.1f}  {s['avg_dropped']:>8.1f}"
              f"  {s['avg_drop_rate']:>9.1%}")


def print_q4_overlap(results: dict) -> None:
    print("\n=== Q4: Martingale-Fragile Cases vs Framing/Info Harm Cases ===\n")
    print("  Low overlap → ordering-sensitivity and persuasion harms are independent risks.\n")
    hdr = (f"{'Agent':<12}  {'fragile':>7}  {'frag∩framing':>12}"
           f"  {'frag∩info':>10}  {'fragile_only':>12}")
    print(hdr)
    print("-" * len(hdr))
    for agent in AGENTS:
        m  = results[agent].get("martingale")
        ov = results[agent].get("overlap", {})
        if not m:
            print(f"{agent:<12}  N/A"); continue
        print(f"{agent:<12}  {m['n_fragile']:>7}"
              f"  {ov.get('fragile_and_framing_harm', 0):>12}"
              f"  {ov.get('fragile_and_info_harm', 0):>10}"
              f"  {ov.get('fragile_only', 0):>12}")


def print_detail(agent: str) -> None:
    mart_entries = load_martingale(None if agent == "(base)" else agent)
    if not mart_entries:
        print(f"No martingale data for {agent}"); return
    records = per_case_stats(mart_entries)
    fragile = sorted([r for r in records if r["fragile"]], key=lambda x: x["consistency"])
    print(f"\n=== DETAIL: {agent} — {len(fragile)} fragile cases ===\n")
    print(f"{'case_id':<22}  {'perm0':>5}  {'majority':>8}  {'gt':>4}  {'consist':>7}  {'bel_std':>7}  result")
    print("-" * 75)
    for r in fragile:
        outcome = ("recovered"    if not r["perm0_correct"] and r["majority_correct"]  else
                   "degraded"     if r["perm0_correct"]     and not r["majority_correct"] else
                   "stable-wrong" if not r["perm0_correct"] and not r["majority_correct"] else
                   "stable-correct")
        print(f"{r['case_id']:<22}  {r['perm0_answer']:>5}  {r['majority_answer']:>8}"
              f"  {r['gt']:>4}  {r['consistency']:>7.3f}  {r['std_belief']:>7.4f}  {outcome}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, metavar="PATH",
                        help="Output JSON path")
    parser.add_argument("--detail", metavar="AGENT",
                        help="Per-case fragile breakdown for a specific agent")
    args = parser.parse_args()

    results = build_results()

    print_summary_table(results)
    print_cross_condition_table(results)
    print_q1_permutation_benefit(results)
    print_q2_framing_effect(results)
    print_q3_info_design(results)
    print_q4_overlap(results)

    if args.detail:
        print_detail(args.detail)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
