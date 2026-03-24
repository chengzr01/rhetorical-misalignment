#!/usr/bin/env python3
"""Aggregate factualness analysis results from multiple agents on the same cases.

For each case that appears across the specified agents, combine the claims from
all agents, with an option to exclude unfactual claims.

Usage examples:
  # Aggregate llama variants, keep all claims
  python aggregate_information.py --agents llama-small llama-dpo llama-sft

  # Aggregate and drop unfactual claims, only cases present in ALL agents
  python aggregate_information.py --agents llama-small llama-dpo llama-sft \\
      --exclude-unfactual --require-all --output aggregated.json

  # Aggregate all available agents, save output
  python aggregate_information.py --agents llama-small llama-dpo llama-sft gpt \\
      --output information/aggregated_llama_variants.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

_SCRIPT_DIR = Path(__file__).parent
INFORMATION_DIR = str(_SCRIPT_DIR / "information")
AGGREGATION_DIR = str(_SCRIPT_DIR / "aggregation")


def load_factualness_file(information_dir: str, agent_name: str) -> Optional[Dict]:
    """Load factualness results for a given agent name."""
    path = Path(information_dir) / f"factualness_agent_{agent_name}.json"
    if not path.exists():
        print(f"WARNING: No factualness file found for agent '{agent_name}' at {path}")
        return None
    return json.loads(path.read_text())


def get_cases_by_id(data: Dict) -> Dict[str, Dict]:
    """Return a mapping of case_id -> result for successfully analyzed cases."""
    return {
        r["case_id"]: r
        for r in data.get("results", [])
        if r["status"] == "success"
    }


def aggregate_cases(
    agents: List[str],
    information_dir: str,
    keep_labels: Set[str],
    require_all: bool,
) -> Dict:
    """Load and aggregate claims from multiple agents across shared cases.

    Args:
        agents: agent name suffixes (e.g. ["llama-small", "llama-dpo"])
        information_dir: directory containing factualness_agent_*.json files
        keep_labels: set of claim labels to retain (any of "factual", "unfactual", "uncertain")
        require_all: if True, only include cases present in ALL agents;
                     if False, include cases present in ANY agent (union)

    Returns:
        dict with "metadata", "aggregate", and "cases" (keyed by case_id)
    """
    # Load per-agent case maps
    agent_cases: Dict[str, Dict[str, Dict]] = {}
    for agent in agents:
        raw = load_factualness_file(information_dir, agent)
        if raw is None:
            continue
        agent_cases[agent] = get_cases_by_id(raw)

    if not agent_cases:
        print("ERROR: No agent data loaded.")
        return {}

    loaded_agents = list(agent_cases.keys())

    # Determine which case IDs to include
    all_case_ids: Set[str] = set()
    for cases in agent_cases.values():
        all_case_ids |= set(cases.keys())

    if require_all:
        case_ids = sorted(
            cid for cid in all_case_ids
            if all(cid in agent_cases[a] for a in loaded_agents)
        )
    else:
        case_ids = sorted(all_case_ids)

    aggregated_cases: Dict[str, Dict] = {}
    total_claims_before = 0
    total_claims_after = 0

    for case_id in case_ids:
        combined_claims: List[Dict] = []
        agents_present: List[str] = []

        for agent in loaded_agents:
            if case_id not in agent_cases[agent]:
                continue
            agents_present.append(agent)
            result = agent_cases[agent][case_id]

            for claim in result["analysis"].get("claims", []):
                total_claims_before += 1
                if claim.get("label") in keep_labels:
                    combined_claims.append({
                        "agent": agent,
                        "claim": claim["claim"],
                        "label": claim["label"],
                        "explanation": claim.get("explanation", ""),
                    })
                    total_claims_after += 1

        label_counts: Dict[str, int] = defaultdict(int)
        for c in combined_claims:
            label_counts[c["label"]] += 1

        aggregated_cases[case_id] = {
            "case_id": case_id,
            "agents_present": agents_present,
            "n_agents": len(agents_present),
            "claims": combined_claims,
            "summary": {
                "total_claims": len(combined_claims),
                "factual_count": label_counts["factual"],
                "unfactual_count": label_counts["unfactual"],
                "uncertain_count": label_counts["uncertain"],
            },
        }

    return {
        "metadata": {
            "agents": loaded_agents,
            "information_dir": str(information_dir),
            "keep_labels": sorted(keep_labels),
            "require_all_agents": require_all,
            "n_agents_loaded": len(loaded_agents),
            "n_cases": len(case_ids),
        },
        "aggregate": {
            "n_cases": len(case_ids),
            "total_claims_before_filter": total_claims_before,
            "total_claims_after_filter": total_claims_after,
            "claims_excluded": total_claims_before - total_claims_after,
        },
        "cases": aggregated_cases,
    }


def print_summary(result: Dict) -> None:
    meta = result["metadata"]
    agg = result["aggregate"]
    cases = result["cases"]

    print("\n" + "=" * 80)
    print("AGGREGATED INFORMATION SUMMARY")
    print("=" * 80)
    print(f"Agents:               {', '.join(meta['agents'])}")
    print(f"Labels included:      {', '.join(meta['keep_labels'])}")
    print(f"Require all agents:   {meta['require_all_agents']}")
    print(f"Cases aggregated:     {agg['n_cases']}")
    print(f"Claims before filter: {agg['total_claims_before_filter']}")
    print(f"Claims after filter:  {agg['total_claims_after_filter']}")
    if agg["claims_excluded"]:
        print(f"Claims excluded:      {agg['claims_excluded']} claims removed (not in kept labels)")
    print("-" * 80)

    preview_n = min(10, len(cases))
    print(f"\nPer-case summary (first {preview_n} of {len(cases)}):")
    header = f"  {'Case ID':<32} {'Agents':>6} {'Total':>6} {'Factual':>8} {'Unfact':>7} {'Uncert':>7}"
    print(header)
    print("  " + "-" * 72)
    for case_id, case in list(cases.items())[:preview_n]:
        s = case["summary"]
        print(
            f"  {case_id:<32} {case['n_agents']:>6} "
            f"{s['total_claims']:>6} "
            f"{s['factual_count']:>8} "
            f"{s['unfactual_count']:>7} "
            f"{s['uncertain_count']:>7}"
        )
    if len(cases) > preview_n:
        print(f"  ... and {len(cases) - preview_n} more cases")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agents", nargs="+", required=True,
        metavar="AGENT",
        help="Agent names to aggregate (e.g. llama-small llama-dpo llama-sft)",
    )
    parser.add_argument(
        "--information-dir", default=INFORMATION_DIR,
        help="Directory containing factualness_agent_*.json files",
    )
    parser.add_argument(
        "--labels", nargs="+",
        choices=["factual", "unfactual", "uncertain"],
        default=["factual"],
        metavar="LABEL",
        help=(
            "Claim labels to include in the aggregation. "
            "Choose any combination of: factual, unfactual, uncertain. "
            "(default: factual)"
        ),
    )
    parser.add_argument(
        "--require-all", action="store_true",
        help=(
            "Only include cases present in ALL specified agents "
            "(default: union of all cases)"
        ),
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Save aggregated JSON to this path (default: aggregation/aggregated_<labels>.json)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the printed summary",
    )

    args = parser.parse_args()

    labels = set(args.labels)
    result = aggregate_cases(
        agents=args.agents,
        information_dir=args.information_dir,
        keep_labels=labels,
        require_all=args.require_all,
    )

    if not result:
        return

    if not args.quiet:
        print_summary(result)

    label_tag = "_".join(sorted(labels))
    out_path = Path(args.output) if args.output else (
        Path(AGGREGATION_DIR) / f"aggregated_{label_tag}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nAggregated output saved → {out_path}")


if __name__ == "__main__":
    main()
