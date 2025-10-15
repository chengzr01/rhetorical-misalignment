from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

import yaml

from experiments.hypothesis import HypothesisGenerator
from experiments.verification import DataVerifier
from interface.client import OpenRouterChatClient, SGLangChatClient, NvidiaChatClient


def setup_client(backend: str) -> OpenRouterChatClient | SGLangChatClient:
    if backend == "nvidia":
        return NvidiaChatClient()
    elif backend == "openrouter":
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


def get_context_template(
    template_path: str | Path = "prompts/experiments/context_template.yaml",
) -> str:
    with open(template_path, "r") as f:
        template_config = yaml.safe_load(f)
        return template_config["template"]


def get_agent_configs() -> list[dict]:
    return [
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
    ]


def get_principal_configs() -> list[dict]:
    return [
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
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="nvidia")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-v3.1")
    parser.add_argument("--data-dir", type=str, default="datasets/processed")
    parser.add_argument(
        "--output", type=str, default="experiments/input/hypothesis.json"
    )
    parser.add_argument("--n-hypotheses", type=int, default=100)
    parser.add_argument("--age-window", type=int, default=10)
    args = parser.parse_args()

    client = setup_client(args.server)
    generator = HypothesisGenerator(args.data_dir, client, args.model)
    verifier = DataVerifier(args.data_dir)

    drug_lab_pairs = generator.get_drug_lab_pairs(min_episodes=15)
    print(f"Found {len(drug_lab_pairs)} drug-lab pairs with sufficient data")

    contexts = []
    count = 0

    for drug, lab_test in tqdm(drug_lab_pairs[: args.n_hypotheses]):
        print(f"Generating hypothesis for drug {drug} and lab test {lab_test}")
        patient_ids = generator.select_patients_for_drug(drug, n_patients=1)
        print(f"Selected {len(patient_ids)} patients for drug {drug}")

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

        context_text = get_context_template().format(
            patient_id=patient_info["patient_id"],
            sex=patient_info["sex"],
            age=patient_info["age"],
            drug=drug,
            hypothesis=hypothesis,
            data_report=data_report,
        )

        contexts.append(
            {
                "text": context_text,
                "patient_id": patient_info["patient_id"],
                "sex": patient_info["sex"],
                "age": patient_info["age"],
                "drug": drug,
                "lab_test": lab_test,
                "hypothesis": hypothesis,
                "data_report": data_report,
                "age_range": age_range,
            }
        )
        count += 1

        if count >= args.n_hypotheses:
            break

    config = {
        "contexts": contexts,
        "agent_configs": get_agent_configs(),
        "principal_configs": get_principal_configs(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Generated {len(contexts)} hypothesis-verification pairs")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
