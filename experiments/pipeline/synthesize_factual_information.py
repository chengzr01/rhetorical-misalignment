#!/usr/bin/env python3
"""
Synthesize per-model factual/unfactual agent output files from factualness analysis.

For each model that has a factualness analysis in experiments/information/,
this script produces two files in experiments/agents/usmle_sample/:

  agent_{model}_factual.json   -- information field contains only "factual" claims
  agent_{model}_unfactual.json -- information field contains only "unfactual" +
                                   "uncertain" claims

The output format matches the framing output of agent_presentation.py so these
files can be passed directly to principal_inference.py via --agent-cache.

Usage:
    python experiments/synthesize_factual_information.py
    python experiments/synthesize_factual_information.py --models claude gpt gemini
    python experiments/synthesize_factual_information.py --claim-format numbered
    python experiments/synthesize_factual_information.py --agents-dir experiments/agents/usmle_sample
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


INFORMATION_DIR = Path("experiments/information")
AGENTS_DIR = Path("experiments/agents/usmle_sample")

# Labels that count as "unfactual" in the unfactual split
UNFACTUAL_LABELS = {"unfactual", "uncertain"}
FACTUAL_LABELS = {"factual"}


def format_claims(claims: list[dict[str, Any]], fmt: str) -> str:
    """Join a list of claim dicts into a single information string."""
    lines: list[str] = []
    for i, c in enumerate(claims, start=1):
        text = c.get("claim", "").strip()
        if not text:
            continue
        if fmt == "numbered":
            lines.append(f"{i}. {text}")
        elif fmt == "bullets":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def synthesize_model(
    model_key: str,
    factualness_path: Path,
    agents_dir: Path,
    claim_format: str,
    force: bool,
) -> tuple[int, int]:
    """
    Synthesize factual and unfactual agent files for one model.

    Returns:
        (n_factual_records, n_unfactual_records) written
    """
    agent_path = agents_dir / f"agent_{model_key}.json"
    if not agent_path.exists():
        print(f"  [SKIP] {model_key}: agent file not found at {agent_path}")
        return 0, 0

    factual_out = agents_dir / f"agent_{model_key}_factual.json"
    unfactual_out = agents_dir / f"agent_{model_key}_unfactual.json"
    all_claims_out = agents_dir / f"agent_{model_key}_all_claims.json"

    if not force and factual_out.exists() and unfactual_out.exists() and all_claims_out.exists():
        print(f"  [SKIP] {model_key}: output files already exist (use --force to overwrite)")
        return 0, 0

    # Load factualness results
    factualness = json.loads(factualness_path.read_text())
    fact_results = factualness.get("results", [])

    # Build lookup: case_id → factualness result
    fact_by_id: dict[str, dict] = {}
    for r in fact_results:
        if r.get("status") == "success":
            fact_by_id[r["case_id"]] = r

    # Load original agent records (for metadata)
    agent_records: list[dict] = json.loads(agent_path.read_text())
    agent_by_id: dict[str, dict] = {r["case_id"]: r for r in agent_records}

    factual_records: list[dict] = []
    unfactual_records: list[dict] = []
    all_claims_records: list[dict] = []
    skipped = 0

    for case_id, fact_result in fact_by_id.items():
        orig = agent_by_id.get(case_id)
        if orig is None:
            skipped += 1
            continue

        all_claims: list[dict] = fact_result.get("analysis", {}).get("claims", [])

        factual_claims = [c for c in all_claims if c.get("label") in FACTUAL_LABELS]
        unfactual_claims = [c for c in all_claims if c.get("label") in UNFACTUAL_LABELS]

        factual_info = format_claims(factual_claims, claim_format)
        unfactual_info = format_claims(unfactual_claims, claim_format)
        all_claims_info = format_claims(all_claims, claim_format)

        # Base record — all metadata from the original agent record
        base = {
            "agent_context": orig.get("agent_context") or orig.get("principal_context", ""),
            "principal_context": orig.get("principal_context") or orig.get("agent_context", ""),
            "agent_task": orig.get("agent_task"),
            "agent_objective": orig.get("agent_objective"),
            "case_id": case_id,
            "dataset_type": orig.get("dataset_type", "usmle"),
            "options": orig.get("options"),
            "correct_answer": orig.get("correct_answer"),
            "correct_answer_idx": orig.get("correct_answer_idx"),
            "meta_info": orig.get("meta_info"),
            "ground_truth_agent": orig.get("agent_name"),
            "ground_truth_model": orig.get("agent_model"),
        }

        # Factual record
        factual_records.append({
            "agent_name": f"synthesized_factual_{model_key}",
            "agent_model": orig.get("agent_model", ""),
            **base,
            "provided_information": factual_info,
            "information": factual_info,
            "n_factual_claims": len(factual_claims),
            "n_total_claims": len(all_claims),
        })

        # Unfactual record
        unfactual_records.append({
            "agent_name": f"synthesized_unfactual_{model_key}",
            "agent_model": orig.get("agent_model", ""),
            **base,
            "provided_information": unfactual_info,
            "information": unfactual_info,
            "n_unfactual_claims": len(unfactual_claims),
            "n_total_claims": len(all_claims),
        })

        # All-claims record (factual + unfactual + uncertain combined)
        all_claims_records.append({
            "agent_name": f"synthesized_all_claims_{model_key}",
            "agent_model": orig.get("agent_model", ""),
            **base,
            "provided_information": all_claims_info,
            "information": all_claims_info,
            "n_factual_claims": len(factual_claims),
            "n_unfactual_claims": len(unfactual_claims),
            "n_total_claims": len(all_claims),
        })

    factual_out.write_text(json.dumps(factual_records, indent=2))
    unfactual_out.write_text(json.dumps(unfactual_records, indent=2))
    all_claims_out.write_text(json.dumps(all_claims_records, indent=2))

    if skipped:
        print(f"  [WARN] {model_key}: {skipped} case(s) missing from agent file, skipped")

    print(
        f"  {model_key}: {len(factual_records)} records written "
        f"(factual → {factual_out.name}, "
        f"unfactual → {unfactual_out.name}, "
        f"all_claims → {all_claims_out.name})"
    )
    return len(factual_records), len(unfactual_records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize factual/unfactual agent files from factualness analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        metavar="MODEL",
        help="Model keys to process (default: all found in experiments/information/)",
    )
    parser.add_argument(
        "--information-dir",
        type=str,
        default=str(INFORMATION_DIR),
        help="Directory containing factualness_agent_*.json files",
    )
    parser.add_argument(
        "--agents-dir",
        type=str,
        default=str(AGENTS_DIR),
        help="Directory containing agent_*.json files (also where output is written)",
    )
    parser.add_argument(
        "--claim-format",
        type=str,
        default="bullets",
        choices=["bullets", "numbered", "plain"],
        help="Format for joining claims into the information string",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    info_dir = Path(args.information_dir)
    agents_dir = Path(args.agents_dir)

    # Discover factualness files
    fact_files = sorted(info_dir.glob("factualness_agent_*.json"))
    if not fact_files:
        print(f"No factualness_agent_*.json files found in {info_dir}")
        return

    # Optionally filter by --models
    if args.models:
        wanted = set(args.models)
        fact_files = [f for f in fact_files if f.stem.replace("factualness_agent_", "") in wanted]
        if not fact_files:
            print(f"No matching factualness files for models: {args.models}")
            return

    print(f"Found {len(fact_files)} factualness file(s) to process")
    print(f"Claim format: {args.claim_format}")
    print(f"Output dir:   {agents_dir}\n")

    total_factual = total_unfactual = 0
    for fact_path in fact_files:
        model_key = fact_path.stem.replace("factualness_agent_", "")
        print(f"Processing model: {model_key}")
        nf, nu = synthesize_model(
            model_key=model_key,
            factualness_path=fact_path,
            agents_dir=agents_dir,
            claim_format=args.claim_format,
            force=args.force,
        )
        total_factual += nf
        total_unfactual += nu

    print(f"\nDone. Total records written: {total_factual} factual, {total_unfactual} unfactual")


if __name__ == "__main__":
    main()
