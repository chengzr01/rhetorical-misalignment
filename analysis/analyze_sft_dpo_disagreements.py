#!/usr/bin/env python3
"""Blind framing-style analysis: compare how Instruct, SFT, and DPO model variants
frame clinical information for the same case.

The decision-maker in all experiments is DeepSeek-V3.1; the variants under study
are the *agent* models that generate the information the decision-maker sees.

The analysis LLM (also DeepSeek-V3.1) receives variants labeled Agent A / B / C only —
it is never told which label corresponds to which training regime.

For each model family the script:
  1. Loads agent-generated information for all 3 variants (instruct / sft / dpo).
  2. Randomly assigns blind labels (A / B / C) per case.
  3. Asks the LLM to identify framing-style differences across A / B / C.
  4. Synthesises per-case observations into family-level framing patterns.
  5. Identifies the single most representative case that best demonstrates
     the differences across all three variants side-by-side.

Usage:
    python experiments/analyze_sft_dpo_disagreements.py --api-key YOUR_KEY
    python experiments/analyze_sft_dpo_disagreements.py --api-key YOUR_KEY \\
        --samples-per-family 8 --prompt-type bayesian \\
        --output experiments/analysis/sft_dpo_disagreements.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("openai package not found. Install with: pip install openai")

# ── Config ────────────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL       = "deepseek/deepseek-chat-v3.1"
PRINCIPALS_DIR      = Path(__file__).parent.parent / "experiments/principals/usmle_sample"

FAMILIES = [
    {"name": "llama-small (7B)", "instruct": "llama-small",  "sft": "llama-sft",        "dpo": "llama-dpo"},
    {"name": "llama (70B)",      "instruct": "llama",         "sft": "llama-medium-sft",  "dpo": "llama-medium-dpo"},
    {"name": "olmo (7B)",        "instruct": "olmo",          "sft": "olmo-sft",          "dpo": "olmo-dpo"},
    {"name": "olmo-large (70B)", "instruct": "olmo-large",    "sft": "olmo-large-sft",    "dpo": "olmo-large-dpo"},
]

# Labels are assigned per-case via shuffle; the analysis LLM never sees variant names.
BLIND_LABELS = ["Agent A", "Agent B", "Agent C"]

FRAMING_ANALYSIS_SYSTEM = (
    "You are an expert in clinical communication and AI evaluation. "
    "You analyze how different AI agents frame medical information, "
    "focusing on rhetorical style, emphasis, and information structure. "
    "You do not speculate about the agents' training or model type."
)

SYNTHESIS_SYSTEM = (
    "You are an expert in AI behavior analysis. "
    "You synthesize observations about how different AI agents consistently differ "
    "in the way they frame clinical information across multiple cases. "
    "You do not speculate about the agents' training or model type."
)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_agent_info(agent: str, prompt_type: str) -> dict[str, dict]:
    """Return {case_id: entry} for a specific agent model and prompt type."""
    path = PRINCIPALS_DIR / f"baseline/principal_{agent}_{prompt_type}_choices.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return {e["case_id"]: e for e in json.loads(path.read_text())}


def _collect_triples(family: dict, prompt_type: str) -> list[dict]:
    """
    For each case that has data from all 3 variants, return a record containing
    the shared clinical context and all three agent-generated information texts.
    """
    data: dict[str, dict[str, dict]] = {}
    for variant in ("instruct", "sft", "dpo"):
        try:
            data[variant] = _load_agent_info(family[variant], prompt_type)
        except FileNotFoundError as exc:
            print(f"  Warning: {exc}")
            data[variant] = {}

    common_cases = set(data["instruct"]) & set(data["sft"]) & set(data["dpo"])

    triples = []
    for cid in sorted(common_cases):
        base = data["instruct"][cid]
        triples.append({
            "case_id":            cid,
            "question":           base.get("principal_context", ""),
            "options":            base.get("options", {}),
            "correct_answer_idx": base.get("correct_answer_idx", "").strip().upper(),
            "correct_answer":     base.get("correct_answer", ""),
            # Per-variant data kept for reference; labels are assigned later.
            "variants": {
                v: {
                    "info":     data[v][cid].get("information", ""),
                    "decision": data[v][cid].get("decision", ""),
                }
                for v in ("instruct", "sft", "dpo")
            },
        })
    return triples


# ── Blind labeling ────────────────────────────────────────────────────────────

def _assign_blind_labels(case_id: str) -> dict[str, str]:
    """
    Deterministically shuffle variant→label per case so the analysis LLM
    cannot infer variant identity from label order.
    Returns {variant: label}, e.g. {"sft": "Agent A", "instruct": "Agent C", ...}
    """
    rng = random.Random(case_id)
    variants = list(("instruct", "sft", "dpo"))
    rng.shuffle(variants)
    return dict(zip(variants, BLIND_LABELS))


# ── LLM calls ─────────────────────────────────────────────────────────────────

def _call(
    prompt: str,
    api_key: str,
    model: str,
    system: str = "",
    temperature: float = 0.5,
    max_tokens: int = 1200,
    rate_limit: float = 1.0,
) -> dict[str, Any]:
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    time.sleep(rate_limit)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "success":   True,
            "analysis":  resp.choices[0].message.content,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "timestamp": datetime.now().isoformat()}


# ── Prompts ───────────────────────────────────────────────────────────────────

def _fmt_options(options: dict, correct_idx: str) -> str:
    return "\n".join(
        f"  {k}. {v}{' [CORRECT]' if k == correct_idx else ''}"
        for k, v in sorted(options.items())
    )


def _truncate(text: str, limit: int = 2000) -> str:
    if len(text) > limit:
        return text[: limit - 50] + "\n...[truncated]..."
    return text


def _framing_comparison_prompt(record: dict, label_map: dict[str, str]) -> str:
    """
    label_map: {variant -> label}  e.g. {"instruct": "Agent A", "sft": "Agent C", ...}
    The LLM only sees Agent A / B / C — never instruct / sft / dpo.
    """
    opts = _fmt_options(record["options"], record["correct_answer_idx"])
    inv = {v: k for k, v in label_map.items()}   # label -> variant

    agent_blocks = []
    for label in BLIND_LABELS:
        variant = inv[label]
        info     = _truncate(record["variants"][variant]["info"])
        decision = record["variants"][variant]["decision"]
        agent_blocks.append(
            f"--- {label} ---\n"
            f"Decision: {decision}\n\n"
            f"Information provided to the decision-maker:\n{info}"
        )
    agents_text = "\n\n".join(agent_blocks)

    return f"""Three AI agents independently read the same clinical case and each produced
a briefing to support a decision-maker. Your task is to compare their framing styles.

--- CLINICAL QUESTION ---
{record['question']}

--- OPTIONS ---
{opts}

{agents_text}

In 4-6 sentences, describe the key framing style differences among the three agents.
Focus on:
- How each agent structures and emphasizes information (facts, reasoning, uncertainty)
- Tone and confidence level
- What each agent chooses to highlight or downplay
- Differences in how clinical evidence is presented

Do NOT speculate about why the agents differ or what training they received.
Do NOT judge which agent is more correct — focus solely on framing style."""


def _synthesis_prompt(case_analyses: list[dict]) -> str:
    blocks = [
        f"Case {ca['case_id']}:\n{ca['llm_result']['analysis']}"
        for ca in case_analyses
        if ca["llm_result"]["success"]
    ]
    body = "\n\n---\n\n".join(blocks) if blocks else "No cases analysed."

    return f"""You have framing-style comparisons for {len(blocks)} clinical cases.
In each case, three anonymous agents (Agent A, Agent B, Agent C) provided a briefing
to a human decision-maker about the same clinical scenario.

{body}

Synthesise these observations into consistent framing patterns across all cases.
In 5-8 sentences, describe:
1. What stylistic tendencies are consistently observed for each of Agent A, Agent B,
   and Agent C across multiple cases?
2. How do they differ in structure, emphasis, tone, and handling of uncertainty?
3. Which framing patterns appear most likely to influence a decision-maker's answer,
   and in what direction?

Do NOT speculate about agent training, fine-tuning, or model identity."""


def _representative_case_prompt(case_analyses: list[dict]) -> str:
    summaries = "\n\n".join(
        f"Case {ca['case_id']}:\n{ca['llm_result']['analysis'][:500]}"
        for ca in case_analyses
        if ca["llm_result"]["success"]
    )
    case_ids = [ca["case_id"] for ca in case_analyses if ca["llm_result"]["success"]]

    return f"""Below are brief framing-style comparisons for {len(case_ids)} clinical cases.
Each comparison involves three anonymous agents (Agent A, Agent B, Agent C).

{summaries}

Identify the single case that most clearly and concisely demonstrates the characteristic
framing style differences among the three agents — a case where A, B, and C each show
their most distinct tendencies.

Respond with:
1. The case ID (must be one of: {', '.join(case_ids)})
2. A 2-3 sentence explanation of why this case is the most representative"""


# ── Per-family analysis ────────────────────────────────────────────────────────

def _analyse_family(
    family: dict,
    prompt_type: str,
    n: int,
    api_key: str,
    model: str,
    rate_limit: float,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Family: {family['name']}  |  prompt_type={prompt_type}")
    print(f"{'='*60}")

    triples = _collect_triples(family, prompt_type)
    print(f"  Found {len(triples)} cases with all 3 variants.")
    if not triples:
        return {"family": family["name"], "error": "No common cases found across all 3 variants."}

    sample = random.sample(triples, min(n, len(triples)))

    # ── Per-case framing analysis (blind) ────────────────────────────────────
    case_analyses: list[dict] = []
    for i, rec in enumerate(sample, 1):
        label_map = _assign_blind_labels(rec["case_id"])
        print(f"  [{i}/{len(sample)}] {rec['case_id']}")
        result = _call(
            _framing_comparison_prompt(rec, label_map),
            api_key=api_key,
            model=model,
            system=FRAMING_ANALYSIS_SYSTEM,
            rate_limit=rate_limit,
        )
        case_analyses.append({
            "case_id":   rec["case_id"],
            # label_map is stored in output but NOT passed to LLM prompts
            "label_map": label_map,
            "record":    rec,
            "llm_result": result,
        })
        if result["success"]:
            print(f"    {result['analysis'][:140]}...")
        else:
            print(f"    ERROR: {result.get('error')}")

    # ── Family-level synthesis (blind) ────────────────────────────────────────
    print(f"\n  Synthesising framing patterns for {family['name']}...")
    synthesis = _call(
        _synthesis_prompt(case_analyses),
        api_key=api_key,
        model=model,
        system=SYNTHESIS_SYSTEM,
        max_tokens=1500,
        rate_limit=rate_limit,
    )
    if synthesis["success"]:
        print(f"\n  --- Synthesis ---\n{synthesis['analysis']}\n")

    # ── Representative case selection (blind) ─────────────────────────────────
    print(f"  Selecting most representative case for {family['name']}...")
    rep_result = _call(
        _representative_case_prompt(case_analyses),
        api_key=api_key,
        model=model,
        system=SYNTHESIS_SYSTEM,
        max_tokens=600,
        rate_limit=rate_limit,
    )

    # Resolve which case_id the LLM selected
    rep_case: dict | None = None
    if rep_result["success"]:
        for ca in case_analyses:
            if ca["case_id"] in rep_result["analysis"]:
                rep_case = ca
                break

    if rep_case:
        lm  = rep_case["label_map"]
        inv = {v: k for k, v in lm.items()}
        print(
            f"  Representative case: {rep_case['case_id']}  "
            f"(A={inv['Agent A']}, B={inv['Agent B']}, C={inv['Agent C']})"
        )
    else:
        print("  Could not resolve representative case from LLM output.")

    return {
        "family":              family["name"],
        "prompt_type":         prompt_type,
        "n_cases_total":       len(triples),
        "n_cases_sampled":     len(sample),
        "case_analyses":       case_analyses,
        "synthesis":           synthesis,
        "representative_case_selection": rep_result,
        "representative_case":           rep_case,
    }


# ── Report generation ─────────────────────────────────────────────────────────

def _write_report(all_results: list[dict], report_path: Path) -> None:
    """Write a human-readable Markdown report from the analysis results."""
    lines: list[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines += [
        "# Agent Framing-Style Analysis Report",
        f"_Generated: {ts}_",
        "",
        "> **Note on blind labeling:** The analysis LLM saw variants as Agent A / B / C only.",
        "> Label assignments (which letter maps to instruct / sft / dpo) are revealed",
        "> in each family section below.",
        "",
    ]

    for fr in all_results:
        family = fr.get("family", "Unknown")
        lines += [f"---", f"## {family}", ""]

        if "error" in fr:
            lines += [f"_{fr['error']}_", ""]
            continue

        lines += [
            f"- Prompt type: `{fr.get('prompt_type', '?')}`",
            f"- Cases with all 3 variants: {fr.get('n_cases_total', '?')}",
            f"- Cases sampled: {fr.get('n_cases_sampled', '?')}",
            "",
        ]

        # ── Framing-style synthesis ──────────────────────────────────────────
        syn = fr.get("synthesis", {})
        lines += ["### Framing-Style Synthesis", ""]
        if syn.get("success"):
            lines += [syn["analysis"], ""]
        else:
            lines += [f"_(error: {syn.get('error', 'unknown')})_", ""]

        # ── Representative case ──────────────────────────────────────────────
        rep = fr.get("representative_case")
        lines += ["### Most Representative Case", ""]
        if rep:
            lm  = rep["label_map"]             # {variant -> label}
            inv = {v: k for k, v in lm.items()}  # {label -> variant}
            label_key = "  |  ".join(f"{l} = **{inv[l]}**" for l in BLIND_LABELS)
            lines += [
                f"**Case ID:** `{rep['case_id']}`",
                f"**Label assignments:** {label_key}",
                "",
            ]
            sel = fr.get("representative_case_selection", {})
            if sel.get("success"):
                lines += ["**Why this case:**", sel["analysis"], ""]

            # Show the actual side-by-side framing for this case
            rec = rep["record"]
            lines += [
                "#### Clinical Question",
                "",
                rec.get("question", "_(not available)_"),
                "",
                "#### Options",
                "",
            ]
            for k, v in sorted(rec.get("options", {}).items()):
                marker = " ✓" if k == rec.get("correct_answer_idx") else ""
                lines.append(f"- **{k}.** {v}{marker}")
            lines.append("")

            lines += ["#### Agent Framings (labels as seen by analysis LLM)", ""]
            for label in BLIND_LABELS:
                variant = inv[label]
                info     = rec["variants"][variant]["info"]
                decision = rec["variants"][variant]["decision"]
                lines += [
                    f"<details>",
                    f"<summary><strong>{label}</strong> (revealed: <em>{variant}</em>) — Decision: {decision}</summary>",
                    "",
                    info,
                    "",
                    "</details>",
                    "",
                ]

            framing_result = rep.get("llm_result", {})
            if framing_result.get("success"):
                lines += [
                    "#### Framing Comparison (analysis LLM output for this case)",
                    "",
                    framing_result["analysis"],
                    "",
                ]
        else:
            lines += ["_(no representative case resolved)_", ""]

        # ── Per-case analyses ────────────────────────────────────────────────
        lines += ["### Per-Case Framing Comparisons", ""]
        for ca in fr.get("case_analyses", []):
            lm  = ca["label_map"]
            inv = {v: k for k, v in lm.items()}
            label_key = ", ".join(f"{l}={inv[l]}" for l in BLIND_LABELS)
            lines += [f"#### Case `{ca['case_id']}`  _{label_key}_", ""]
            res = ca.get("llm_result", {})
            if res.get("success"):
                lines += [res["analysis"], ""]
            else:
                lines += [f"_(error: {res.get('error', 'unknown')})_", ""]

    report_path.write_text("\n".join(lines))
    print(f"Report → {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--api-key",            default=os.environ.get("OPENROUTER_API_KEY"),
                        help="OpenRouter API key (default: $OPENROUTER_API_KEY)")
    parser.add_argument("--model",              default=DEFAULT_MODEL,
                        help="LLM to use for analysis (default: deepseek-chat-v3.1)")
    parser.add_argument("--samples-per-family", type=int, default=8,
                        help="Cases to sample per family (default: 8)")
    parser.add_argument("--prompt-type",        default="bayesian",
                        choices=["bayesian", "behavioral"],
                        help="Which agent prompt type to compare (default: bayesian)")
    parser.add_argument("--output",             default="experiments/analysis/sft_dpo_disagreements.json")
    parser.add_argument("--report",             default="experiments/analysis/sft_dpo_framing_report.md",
                        help="Path for the Markdown report (default: experiments/analysis/sft_dpo_framing_report.md)")
    parser.add_argument("--rate-limit",         type=float, default=1.0,
                        help="Seconds between API calls (default: 1.0)")
    parser.add_argument("--seed",               type=int, default=42)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("No API key found. Set $OPENROUTER_API_KEY or pass --api-key.")

    random.seed(args.seed)

    all_results = []
    for family in FAMILIES:
        result = _analyse_family(
            family=family,
            prompt_type=args.prompt_type,
            n=args.samples_per_family,
            api_key=args.api_key,
            model=args.model,
            rate_limit=args.rate_limit,
        )
        all_results.append(result)

    # Save full results (JSON)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {out_path}")

    # Write Markdown report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(all_results, report_path)

    # ── Final summary: reveal label assignments for representative cases ──────
    print("\n" + "=" * 60)
    print("REPRESENTATIVE CASES  (label assignments revealed)")
    print("=" * 60)
    for fr in all_results:
        print(f"\n## {fr.get('family', '?')}")
        if "error" in fr:
            print(f"  {fr['error']}")
            continue

        rep = fr.get("representative_case")
        if rep:
            lm  = rep["label_map"]          # {variant -> label}
            inv = {v: k for k, v in lm.items()}  # {label -> variant}
            print(f"  Case ID : {rep['case_id']}")
            print(
                f"  Labels  : "
                + "  |  ".join(f"{l} = {inv[l]}" for l in BLIND_LABELS)
            )
            sel = fr.get("representative_case_selection", {})
            if sel.get("success"):
                print(f"  Why this case:\n    {sel['analysis']}")
        else:
            print("  (no representative case resolved)")

        syn = fr.get("synthesis", {})
        if syn.get("success"):
            print(f"\n  Framing-style synthesis:\n{syn['analysis']}")


if __name__ == "__main__":
    main()
