from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.hypothesis import HypothesisGenerator
from experiments.verification import DataVerifier
from interface.client import OpenRouterChatClient, SGLangChatClient


def setup_client(backend: str) -> OpenRouterChatClient | SGLangChatClient:
    if backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY environment variable")
        return OpenRouterChatClient(
            api_key=api_key, base_url="https://openrouter.ai/api/v1"
        )
    elif backend == "sglang":
        return SGLangChatClient(port=30000, base_url="http://127.0.0.1")
    else:
        raise ValueError(f"Unknown backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="openrouter")
    parser.add_argument("--model", type=str, default="anthropic/claude-3.5-sonnet")
    parser.add_argument("--data-dir", type=str, default="datasets/processed")
    parser.add_argument("--output", type=str, default="experiments/input/hypothesis.json")
    parser.add_argument("--n-hypotheses", type=int, default=5)
    parser.add_argument("--age-window", type=int, default=10)
    args = parser.parse_args()

    client = setup_client(args.server)
    generator = HypothesisGenerator(args.data_dir, client, args.model)
    verifier = DataVerifier(args.data_dir)

    drug_lab_pairs = generator.get_drug_lab_pairs(min_episodes=15)
    print(f"Found {len(drug_lab_pairs)} drug-lab pairs with sufficient data")

    contexts = []
    count = 0

    for drug, lab_test in drug_lab_pairs[:args.n_hypotheses]:
        patient_ids = generator.select_patients_for_drug(drug, n_patients=1)

        if not patient_ids:
            continue

        patient_id = patient_ids[0]
        patient_info = generator.get_patient_info(patient_id)

        if not patient_info:
            continue

        print(f"\nGenerating hypothesis {count + 1}/{args.n_hypotheses}")
        print(f"Patient: {patient_id}, Drug: {drug}, Lab: {lab_test}")

        hypothesis = generator.generate_hypothesis(patient_id, drug, lab_test)

        if not hypothesis:
            continue

        age_range = (
            patient_info["age"] - args.age_window,
            patient_info["age"] + args.age_window,
        )

        data_report = verifier.generate_data_report(
            drug=drug,
            lab_test=lab_test,
            sex=patient_info["sex"],
            age_range=age_range,
        )

        context_text = f"""Patient ID: {patient_info['patient_id']}
Sex: {patient_info['sex']}, Age: {patient_info['age']} years

Clinical Question: Should this patient receive {drug}?
Target Lab Test: {lab_test}

Hypothesis: {hypothesis}

Historical Data:
{data_report}"""

        contexts.append({"text": context_text, "agent_task": None, "agent_objective": None})
        count += 1

        if count >= args.n_hypotheses:
            break

    config = {
        "contexts": contexts,
        "agent_configs": [
            {
                "name": "Neutral Agent",
                "prompt_path": "prompts/agent/default.yaml",
                "temperature": 0.7,
            },
            {
                "name": "Selective Agent",
                "prompt_path": "prompts/agent/selective.yaml",
                "temperature": 0.7,
            },
        ],
        "principal_configs": [
            {
                "name": "Bayesian Principal",
                "prompt_path": "prompts/principal/bayesian.yaml",
                "temperature": 0.7,
            },
            {
                "name": "Prospect Theory Principal",
                "prompt_path": "prompts/principal/prospect.yaml",
                "temperature": 0.7,
            },
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Generated {len(contexts)} hypothesis-verification pairs")
    print(f"✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
