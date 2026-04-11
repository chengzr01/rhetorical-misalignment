#!/usr/bin/env python3
"""Analyze how model answer accuracy affects persuasion effects on humans.

For each case in persuasion_examples.json, we look up whether the model's
own answer (from experiments/tests/) was correct, then cross-reference with
the human's answer change to examine:

  1. When model is correct:  how often does human follow → correct answer?
  2. When model is wrong:    how often does human follow → wrong answer?
  3. Persuasion "alignment": did the model's recommendation match what it
     persuaded the human to do?
  4. Per-model breakdown of the above.
  5. Contingency table: (model correct/wrong) × (human persuasion type).
"""

import json
from collections import defaultdict
from pathlib import Path

PERSUASION_EXAMPLES = (
    Path(__file__).parent.parent
    / "annotation" / "analysis" / "outputs" / "persuasion_examples.json"
)
TESTS_DIR = Path(__file__).parent.parent / "experiments/tests"
OUTPUT_DIR = Path(__file__).parent.parent / "experiments/analysis"

# Map model name → test file stem
MODEL_TO_TEST_FILE = {
    "allenai/Llama-3.1-Tulu-3-8B-DPO":  "test_usmle_sample_allenai-Llama-3.1-Tulu-3-8B-DPO_belief.json",
    "allenai/Llama-3.1-Tulu-3-8B-SFT":  "test_usmle_sample_allenai-Llama-3.1-Tulu-3-8B-SFT_belief.json",
    "anthropic/claude-haiku-4.5":        "test_usmle_sample_anthropic-claude-haiku-4.5_belief.json",
    "deepseek/deepseek-chat-v3.1":       "test_usmle_sample_deepseek-deepseek-chat-v3.1_belief.json",
    "google/gemini-2.5-pro":             "test_usmle_sample_google-gemini-2.5-pro_belief.json",
    "meta-llama/llama-3.1-405b-instruct":"test_usmle_sample_meta-llama-llama-3.1-405b-instruct_belief.json",
    "meta-llama/llama-3.1-8b-instruct":  "test_usmle_sample_meta-llama-llama-3.1-8b-instruct_belief.json",
    "meta-llama/llama-3.3-70b-instruct": "test_usmle_sample_meta-llama-llama-3.3-70b-instruct_belief.json",
    "openai/gpt-5.1":                    "test_usmle_sample_openai-gpt-5.1_belief.json",
}


def load_test_index() -> dict:
    """Return {model: {case_id: result_dict}} from all test files."""
    index = {}
    for model, fname in MODEL_TO_TEST_FILE.items():
        fpath = TESTS_DIR / fname
        if not fpath.exists():
            print(f"  WARNING: test file not found: {fpath}")
            continue
        data = json.loads(fpath.read_text())
        index[model] = {r["id"]: r for r in data["results"]}
    return index


def load_persuasion_cases():
    data = json.loads(PERSUASION_EXAMPLES.read_text())
    cases = []
    for ptype, lst in data["cases"].items():
        for c in lst:
            cases.append({**c, "persuasion_type": ptype})
    return cases


def pct(n, d):
    return f"{n/d:.1%}" if d else "n/a"


def stats_block(rows, label=""):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    # How often human changed to correct answer
    changed_to_correct = sum(1 for r in rows if r["human_changed_to_correct"])
    changed_to_wrong   = sum(1 for r in rows if not r["human_changed_to_correct"])
    # How often model and human ended on same answer
    model_human_agree  = sum(1 for r in rows if r["model_correct"] == r["human_step2_correct"])
    # Model followed: human changed to the answer implied by model
    human_followed_model = sum(1 for r in rows if r["human_followed_model"])
    return {
        "n": n,
        "changed_to_correct": changed_to_correct,
        "changed_to_wrong": changed_to_wrong,
        "pct_correct": changed_to_correct / n,
        "pct_wrong": changed_to_wrong / n,
        "human_followed_model": human_followed_model,
        "pct_followed_model": human_followed_model / n,
        "model_human_agree": model_human_agree,
        "pct_model_human_agree": model_human_agree / n,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_index = load_test_index()
    persuasion_cases = load_persuasion_cases()

    # ── Build enriched rows ────────────────────────────────────────────────
    rows = []
    missing = []
    for c in persuasion_cases:
        model    = c["model"]
        case_id  = c["case_id"]
        ptype    = c["persuasion_type"]

        model_cases = test_index.get(model, {})
        test_result = model_cases.get(case_id)
        if test_result is None:
            missing.append((case_id, model))
            continue

        model_correct        = test_result["correct"]   # bool or None
        model_predicted_idx  = test_result.get("predicted_answer_idx")
        correct_answer_idx   = c["correct_answer_idx"]

        human_step1_correct  = c["step1_correct"]
        human_step2_correct  = c["step2_correct"]
        human_step2_answer   = c["step2_answer"]
        human_changed_to_correct = human_step2_correct

        # Did human end up on the same answer as the model?
        human_followed_model = (
            model_predicted_idx is not None
            and human_step2_answer == model_predicted_idx
        )

        # Model endorsed the correct answer (model_correct)
        # and human changed to match model → correct change
        rows.append({
            "case_id":               case_id,
            "persuasion_type":       ptype,
            "model":                 model,
            "model_correct":         bool(model_correct) if model_correct is not None else None,
            "model_predicted_idx":   model_predicted_idx,
            "correct_answer_idx":    correct_answer_idx,
            "human_step1_correct":   human_step1_correct,
            "human_step2_correct":   human_step2_correct,
            "human_step2_answer":    human_step2_answer,
            "human_changed_to_correct": human_changed_to_correct,
            "human_followed_model":  human_followed_model,
            "belief_change":         c.get("belief_change", 0),
            "step1_belief":          c.get("step1_belief", 0),
            "step2_belief":          c.get("step2_belief", 0),
        })

    if missing:
        print(f"  Missing test results: {len(missing)}")
        for cid, m in missing[:5]:
            print(f"    {cid} / {m}")

    # ── Overall ────────────────────────────────────────────────────────────
    valid = [r for r in rows if r["model_correct"] is not None]
    model_correct_rows   = [r for r in valid if r["model_correct"]]
    model_incorrect_rows = [r for r in valid if not r["model_correct"]]

    s_correct   = stats_block(model_correct_rows)
    s_incorrect = stats_block(model_incorrect_rows)
    s_all       = stats_block(valid)

    # ── By persuasion type × model correctness ─────────────────────────────
    cell = defaultdict(list)
    for r in valid:
        cell[(r["persuasion_type"], r["model_correct"])].append(r)

    # ── Per-model breakdown ────────────────────────────────────────────────
    by_model = defaultdict(list)
    for r in valid:
        by_model[r["model"]].append(r)

    model_stats = {}
    for model, group in sorted(by_model.items()):
        mc = [r for r in group if r["model_correct"]]
        mi = [r for r in group if not r["model_correct"]]
        model_stats[model] = {
            "n_total": len(group),
            "n_model_correct": len(mc),
            "n_model_incorrect": len(mi),
            "pct_model_correct": len(mc) / len(group) if group else 0,
            "when_model_correct":   stats_block(mc),
            "when_model_incorrect": stats_block(mi),
        }

    # ── Belief change by model correctness ────────────────────────────────
    def mean_belief_change(subset):
        vals = [r["belief_change"] for r in subset]
        return sum(vals) / len(vals) if vals else 0.0

    # ── Write JSON ─────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "total_persuasion_cases": len(persuasion_cases),
            "matched_with_test_results": len(valid),
            "missing": len(missing),
        },
        "overall": {
            "all":             s_all,
            "model_correct":   s_correct,
            "model_incorrect": s_incorrect,
            "mean_belief_change_when_model_correct":   mean_belief_change(model_correct_rows),
            "mean_belief_change_when_model_incorrect": mean_belief_change(model_incorrect_rows),
        },
        "contingency": {
            f"{ptype}_model_correct_{mc}": stats_block(cell[(ptype, mc)])
            for ptype in ("harmful_persuasion", "helpful_persuasion")
            for mc in (True, False)
        },
        "by_model": model_stats,
        "rows": rows,
    }
    out_json = OUTPUT_DIR / "model_accuracy_persuasion.json"
    out_json.write_text(json.dumps(output, indent=2))

    # ── Human-readable report ──────────────────────────────────────────────
    lines = []
    lines += [
        "=" * 80,
        "MODEL ANSWER ACCURACY vs. HUMAN PERSUASION EFFECTS",
        "=" * 80,
        f"\nDataset: {len(persuasion_cases)} persuasion cases, "
        f"{len(valid)} matched with model test results",
        "",
    ]

    # Main contingency table
    lines += [
        "─" * 80,
        "PART 1: CONTINGENCY — MODEL CORRECTNESS × PERSUASION OUTCOME",
        "─" * 80,
        "",
        f"  {'':30} {'n':>5}  {'→Correct':>10}  {'→Wrong':>8}  {'Followed model':>15}",
        "  " + "-" * 72,
    ]
    for label, subset in [
        ("Model CORRECT   (all)",   model_correct_rows),
        ("Model INCORRECT (all)",   model_incorrect_rows),
    ]:
        s = stats_block(subset)
        lines.append(
            f"  {label:30} {s['n']:>5}  "
            f"{s['changed_to_correct']:>4}({pct(s['changed_to_correct'],s['n']):>6})  "
            f"{s['changed_to_wrong']:>3}({pct(s['changed_to_wrong'],s['n']):>6})  "
            f"{s['human_followed_model']:>4}({pct(s['human_followed_model'],s['n']):>8})"
        )

    lines += ["", "  By persuasion type:"]
    for ptype in ("harmful_persuasion", "helpful_persuasion"):
        for mc, mc_label in [(True, "model correct"), (False, "model wrong")]:
            s = stats_block(cell[(ptype, mc)])
            if s["n"] == 0:
                continue
            lines.append(
                f"    [{ptype[:7]}] {mc_label:15} n={s['n']:>3}  "
                f"→correct={s['changed_to_correct']:>2}({pct(s['changed_to_correct'],s['n']):>6})  "
                f"→wrong={s['changed_to_wrong']:>2}({pct(s['changed_to_wrong'],s['n']):>6})  "
                f"followed_model={pct(s['human_followed_model'],s['n']):>6}"
            )

    lines += [
        "",
        f"  Mean belief change when model correct  : "
        f"{mean_belief_change(model_correct_rows):+.3f}",
        f"  Mean belief change when model incorrect: "
        f"{mean_belief_change(model_incorrect_rows):+.3f}",
    ]

    # Per-model table
    lines += [
        "",
        "─" * 80,
        "PART 2: PER-MODEL BREAKDOWN",
        "─" * 80,
        "",
        f"  {'Model':<48} {'n':>4}  {'Mdl%corr':>9}  "
        f"{'→corr|mdl✓':>11}  {'→corr|mdl✗':>11}  {'Δ belief|mdl✓':>14}  {'Δ belief|mdl✗':>14}",
        "  " + "-" * 115,
    ]
    for model, ms in model_stats.items():
        sc = ms["when_model_correct"]
        si = ms["when_model_incorrect"]
        mc_rows = [r for r in by_model[model] if r["model_correct"]]
        mi_rows = [r for r in by_model[model] if not r["model_correct"]]
        lines.append(
            f"  {model:<48} {ms['n_total']:>4}  "
            f"{ms['pct_model_correct']:>9.1%}  "
            f"{pct(sc.get('changed_to_correct',0), sc.get('n',0)):>11}  "
            f"{pct(si.get('changed_to_correct',0), si.get('n',0)):>11}  "
            f"{mean_belief_change(mc_rows):>+14.3f}  "
            f"{mean_belief_change(mi_rows):>+14.3f}"
        )

    # Highlight cases where model was wrong but human followed it anyway
    lines += [
        "",
        "─" * 80,
        "PART 3: CASES WHERE MODEL WAS WRONG AND HUMAN FOLLOWED IT",
        "  (Harmful: human was correct but model led them astray)",
        "─" * 80,
        "",
    ]
    wrong_followed = [
        r for r in valid
        if not r["model_correct"]
        and r["human_followed_model"]
        and r["persuasion_type"] == "harmful_persuasion"
    ]
    lines.append(
        f"  Model wrong + human followed (harmful persuasion): {len(wrong_followed)} cases"
    )
    for r in wrong_followed[:15]:
        lines.append(
            f"    {r['case_id']}  model={r['model_predicted_idx']}  "
            f"correct={r['correct_answer_idx']}  "
            f"human_s1={r['human_step1_correct']}→s2={r['human_step2_correct']}  "
            f"model={r['model']}"
        )

    # Cases where model was correct but human didn't follow
    lines += [
        "",
        "─" * 80,
        "PART 4: CASES WHERE MODEL WAS CORRECT BUT HUMAN DID NOT FOLLOW",
        "  (Missed correction: model had right answer but human still ended wrong)",
        "─" * 80,
        "",
    ]
    correct_not_followed = [
        r for r in valid
        if r["model_correct"]
        and not r["human_followed_model"]
        and r["persuasion_type"] == "helpful_persuasion"
        and not r["human_step2_correct"]
    ]
    lines.append(
        f"  Model correct + human didn't follow (helpful, still wrong): "
        f"{len(correct_not_followed)} cases"
    )
    for r in correct_not_followed[:10]:
        lines.append(
            f"    {r['case_id']}  model→{r['model_predicted_idx']}  "
            f"human→{r['human_step2_answer']}  correct={r['correct_answer_idx']}  "
            f"model={r['model']}"
        )

    out_txt = OUTPUT_DIR / "model_accuracy_persuasion_report.txt"
    out_txt.write_text("\n".join(lines) + "\n")
    print(f"[report] saved → {out_txt}")
    print(f"[json]   saved → {out_json}")

    # Print report to stdout
    print()
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
