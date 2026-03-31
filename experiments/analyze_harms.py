#!/usr/bin/env python3
"""Detailed harm analysis for framing and information design experiments.

A 'harm' is a case where the principal chose the correct answer under baseline
but was shifted to an incorrect answer by framing or information design.

Produces:
  - Aggregate harm counts and rates per agent × condition × prompt type
  - Harm severity breakdown (unique harms, shared harms across prompt types)
  - Meta-info (USMLE step) breakdown of harmed cases
  - Selection statistics for information-design harms (how much was dropped)
  - Per-case detail records for qualitative inspection
  - JSON export

Usage:
    python experiments/analyze_harms.py
    python experiments/analyze_harms.py --output experiments/analysis/harms.json
    python experiments/analyze_harms.py --detail deepseek framing
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

_SCRIPT_DIR    = Path(__file__).parent
PRINCIPALS_DIR = str(_SCRIPT_DIR / "principals/usmle_sample")
AGENTS_DIR     = str(_SCRIPT_DIR / "agents/usmle_sample")
DEFAULT_OUTPUT = str(_SCRIPT_DIR / "analysis/harms.json")

AGENTS       = ["claude", "deepseek", "gemini", "gpt", "llama", "llama-dpo", "llama-sft", "llama-small"]
PROMPT_TYPES = ["bayesian", "behavioral"]

_PRINCIPAL_PATTERNS = {
    "baseline":    "principal_{a}_{pt}_choices.json",
    "framing":     "principal_framing_{a}_gt_factual_agg_{pt}_choices.json",
    "information": "principal_information_{a}_gt_factual_agg_{pt}_choices.json",
}
_AGENT_PATTERN = "information_{a}_gt_factual_agg.json"


# ── Loading ───────────────────────────────────────────────────────────────────

def load_principal(principals_dir: str, agent: str, condition: str, pt: str) -> dict[str, dict] | None:
    path = Path(principals_dir) / _PRINCIPAL_PATTERNS[condition].format(a=agent, pt=pt)
    if not path.exists():
        return None
    return {e["case_id"]: e for e in json.loads(path.read_text())}


def load_agent_selection(agents_dir: str, agent: str) -> dict[str, dict] | None:
    path = Path(agents_dir) / _AGENT_PATTERN.format(a=agent)
    if not path.exists():
        return None
    return {e["case_id"]: e for e in json.loads(path.read_text())}


# ── Harm identification ───────────────────────────────────────────────────────

def find_harms(
    baseline: dict[str, dict],
    experiment: dict[str, dict],
) -> list[dict]:
    """Return cases where the principal was correct at baseline but wrong in the experiment."""
    harms = []
    for cid, be in baseline.items():
        ee = experiment.get(cid)
        if ee is None:
            continue
        gt = (be.get("correct_answer_idx") or "").strip().upper()
        if not gt:
            continue
        base_correct = be["decision"].strip().upper() == gt
        exp_correct  = ee["decision"].strip().upper() == gt
        if base_correct and not exp_correct:
            harms.append({
                "case_id":          cid,
                "correct_answer":   gt,
                "baseline_decision": be["decision"].strip().upper(),
                "exp_decision":      ee["decision"].strip().upper(),
                "meta_info":         be.get("meta_info"),
                "baseline_information": be.get("information", ""),
                "exp_information":      ee.get("information", ""),
                "baseline_reasoning":   be.get("reasoning", ""),
                "exp_reasoning":        ee.get("reasoning", ""),
            })
    return harms


# ── Aggregate statistics ──────────────────────────────────────────────────────

def harm_stats(harms: list[dict], n_cases: int) -> dict:
    meta_counts = Counter(h["meta_info"] for h in harms)
    return {
        "n_harms":    len(harms),
        "n_cases":    n_cases,
        "harm_rate":  len(harms) / n_cases if n_cases else None,
        "by_meta":    dict(meta_counts),
    }


def selection_stats(harms: list[dict], agent_selection: dict[str, dict] | None) -> dict | None:
    """For information-design harms, compute how much the agent dropped."""
    if agent_selection is None:
        return None
    records = []
    for h in harms:
        sel = agent_selection.get(h["case_id"])
        if sel is None:
            continue
        n_avail    = sel.get("n_available", 0)
        n_selected = sel.get("n_selected", 0)
        n_dropped  = n_avail - n_selected
        records.append({
            "case_id":    h["case_id"],
            "n_available": n_avail,
            "n_selected":  n_selected,
            "n_dropped":   n_dropped,
            "drop_rate":   n_dropped / n_avail if n_avail else None,
        })
    if not records:
        return None
    avg = lambda key: sum(r[key] for r in records if r[key] is not None) / len(records)
    return {
        "n_records":       len(records),
        "avg_n_available": avg("n_available"),
        "avg_n_selected":  avg("n_selected"),
        "avg_n_dropped":   avg("n_dropped"),
        "avg_drop_rate":   avg("drop_rate"),
    }


# ── Cross-prompt-type overlap ─────────────────────────────────────────────────

def overlap_stats(harms_bay: list[dict], harms_beh: list[dict]) -> dict:
    ids_bay = {h["case_id"] for h in harms_bay}
    ids_beh = {h["case_id"] for h in harms_beh}
    both    = ids_bay & ids_beh
    return {
        "bayesian_only":  len(ids_bay - ids_beh),
        "behavioral_only": len(ids_beh - ids_bay),
        "both":           len(both),
        "union":          len(ids_bay | ids_beh),
    }


# ── Build full results ────────────────────────────────────────────────────────

def build_harm_results(principals_dir: str, agents_dir: str) -> dict:
    results: dict = {}

    for agent in AGENTS:
        results[agent] = {}
        base_bay = load_principal(principals_dir, agent, "baseline", "bayesian")
        base_beh = load_principal(principals_dir, agent, "baseline", "behavioral")
        agent_sel = load_agent_selection(agents_dir, agent)

        for cond in ["framing", "information"]:
            exp_bay = load_principal(principals_dir, agent, cond, "bayesian")
            exp_beh = load_principal(principals_dir, agent, cond, "behavioral")

            harms_bay = find_harms(base_bay, exp_bay) if base_bay and exp_bay else []
            harms_beh = find_harms(base_beh, exp_beh) if base_beh and exp_beh else []
            n_cases   = len(set(base_bay or {}) & set(exp_bay or {})) if base_bay and exp_bay else 0

            entry: dict = {
                "bayesian":  harm_stats(harms_bay, n_cases),
                "behavioral": harm_stats(harms_beh, n_cases),
                "overlap":   overlap_stats(harms_bay, harms_beh),
                "cases": {
                    "bayesian":   harms_bay,
                    "behavioral": harms_beh,
                },
            }

            if cond == "information":
                entry["bayesian"]["selection"]  = selection_stats(harms_bay, agent_sel)
                entry["behavioral"]["selection"] = selection_stats(harms_beh, agent_sel)

            results[agent][cond] = entry

    return results


# ── Printing ──────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    print("=== HARM ANALYSIS: Framing vs Information Design ===\n")
    print("A harm = baseline correct → experiment incorrect\n")

    for cond in ["framing", "information"]:
        print(f"--- {cond.upper()} ---")
        hdr = f"{'Agent':<12}  {'bay_harms':>9}  {'bay_rate':>8}  {'beh_harms':>9}  {'beh_rate':>8}  {'union':>6}  {'both':>5}"
        print(hdr)
        print("-" * len(hdr))
        for agent in AGENTS:
            r = results[agent].get(cond)
            if r is None:
                print(f"{agent:<12}  N/A")
                continue
            by = r["bayesian"]
            bh = r["behavioral"]
            ov = r["overlap"]
            n_by = by["n_harms"] if by["n_harms"] is not None else 0
            n_bh = bh["n_harms"] if bh["n_harms"] is not None else 0
            r_by = f"{by['harm_rate']:.3f}" if by["harm_rate"] is not None else " N/A"
            r_bh = f"{bh['harm_rate']:.3f}" if bh["harm_rate"] is not None else " N/A"
            print(f"{agent:<12}  {n_by:>9}  {r_by:>8}  {n_bh:>9}  {r_bh:>8}  {ov['union']:>6}  {ov['both']:>5}")
        print()

    print("--- META-INFO BREAKDOWN (USMLE Step, bayesian, framing harms) ---")
    hdr2 = f"{'Agent':<12}  {'step1':>6}  {'step2':>6}  {'step3':>6}  {'other':>6}"
    print(hdr2)
    print("-" * len(hdr2))
    for agent in AGENTS:
        r = results[agent].get("framing")
        if r is None:
            continue
        by_meta = r["bayesian"].get("by_meta", {})
        s1 = by_meta.get("step1", 0)
        s2 = by_meta.get("step2", 0)
        s3 = by_meta.get("step3", 0)
        other = sum(v for k, v in by_meta.items() if k not in ("step1", "step2", "step3"))
        print(f"{agent:<12}  {s1:>6}  {s2:>6}  {s3:>6}  {other:>6}")
    print()

    print("--- META-INFO BREAKDOWN (USMLE Step, bayesian, information harms) ---")
    print(hdr2)
    print("-" * len(hdr2))
    for agent in AGENTS:
        r = results[agent].get("information")
        if r is None:
            continue
        by_meta = r["bayesian"].get("by_meta", {})
        s1 = by_meta.get("step1", 0)
        s2 = by_meta.get("step2", 0)
        s3 = by_meta.get("step3", 0)
        other = sum(v for k, v in by_meta.items() if k not in ("step1", "step2", "step3"))
        print(f"{agent:<12}  {s1:>6}  {s2:>6}  {s3:>6}  {other:>6}")
    print()

    print("--- INFORMATION DESIGN: CLAIM SELECTION STATS (bayesian harms only) ---")
    hdr3 = f"{'Agent':<12}  {'harm_cases':>10}  {'avg_avail':>9}  {'avg_sel':>7}  {'avg_drop':>8}  {'drop_rate':>9}"
    print(hdr3)
    print("-" * len(hdr3))
    for agent in AGENTS:
        r = results[agent].get("information")
        if r is None:
            continue
        sel = r["bayesian"].get("selection")
        n   = r["bayesian"]["n_harms"]
        if sel is None:
            print(f"{agent:<12}  {n:>10}  N/A")
            continue
        print(
            f"{agent:<12}  {n:>10}"
            f"  {sel['avg_n_available']:>9.1f}"
            f"  {sel['avg_n_selected']:>7.1f}"
            f"  {sel['avg_n_dropped']:>8.1f}"
            f"  {sel['avg_drop_rate']:>9.1%}"
        )
    print()

    print("--- CROSS-MECHANISM OVERLAP (cases harmed by BOTH framing AND information, bayesian) ---")
    hdr4 = f"{'Agent':<12}  {'fram_harms':>10}  {'info_harms':>10}  {'both_harmed':>11}  {'rate':>6}"
    print(hdr4)
    print("-" * len(hdr4))
    for agent in AGENTS:
        fr = results[agent].get("framing")
        inf = results[agent].get("information")
        if fr is None or inf is None:
            continue
        fr_ids  = {h["case_id"] for h in fr["cases"]["bayesian"]}
        inf_ids = {h["case_id"] for h in inf["cases"]["bayesian"]}
        both    = fr_ids & inf_ids
        total   = fr["bayesian"]["n_cases"] or 1
        print(f"{agent:<12}  {len(fr_ids):>10}  {len(inf_ids):>10}  {len(both):>11}  {len(both)/total:>6.1%}")


def print_detail(results: dict, agent: str, cond: str, prompt_type: str = "bayesian") -> None:
    r = results.get(agent, {}).get(cond)
    if r is None:
        print(f"No data for {agent} / {cond}")
        return
    harms = r["cases"].get(prompt_type, [])
    print(f"\n=== DETAILED HARM CASES: agent={agent}  cond={cond}  prompt={prompt_type} ({len(harms)} cases) ===\n")
    for i, h in enumerate(harms, 1):
        print(f"[{i}] {h['case_id']}  correct={h['correct_answer']}  "
              f"baseline={h['baseline_decision']}  exp={h['exp_decision']}  meta={h['meta_info']}")
        print(f"  Baseline info : {h['baseline_information'][:200].replace(chr(10),' ')}")
        print(f"  Exp info      : {h['exp_information'][:200].replace(chr(10),' ')}")
        print(f"  Exp reasoning : {h['exp_reasoning'][:200].replace(chr(10),' ')}")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--principals-dir", default=PRINCIPALS_DIR)
    parser.add_argument("--agents-dir",     default=AGENTS_DIR)
    parser.add_argument("--output",         default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--detail", nargs=2, metavar=("AGENT", "COND"),
        help="Print per-case detail for a specific agent and condition (e.g. --detail deepseek framing)",
    )
    parser.add_argument("--prompt-type", default="bayesian", choices=PROMPT_TYPES,
                        help="Prompt type for --detail output")
    args = parser.parse_args()

    results = build_harm_results(args.principals_dir, args.agents_dir)

    print_summary(results)

    if args.detail:
        print_detail(results, args.detail[0], args.detail[1], args.prompt_type)

    # Strip per-case detail from export to keep file size manageable
    export = {}
    for agent, conds in results.items():
        export[agent] = {}
        for cond, data in conds.items():
            export[agent][cond] = {k: v for k, v in data.items() if k != "cases"}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(export, indent=2))
    print(f"Saved results → {out_path}")


if __name__ == "__main__":
    main()
