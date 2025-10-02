import os
import argparse
import json
import yaml
from tqdm import tqdm

from client import OpenRouterChatClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="prompts/check.yaml")
    parser.add_argument("--model", type=str, default="deepseek/deepseek-chat-v3.1")
    parser.add_argument("--output", type=str, default="verified")
    args = parser.parse_args()

    with open(args.results, "r") as f:
        results = json.load(f)

    with open(args.prompt, "r") as f:
        prompt_template = yaml.safe_load(f)["prompt"]

    client = OpenRouterChatClient(api_key=os.getenv("OPENROUTER_API_KEY"))

    verified_results = []
    os.makedirs(args.output, exist_ok=True)

    for i, item in enumerate(tqdm(results)):
        context = str(item["params"]["public"])
        advice = item.get("response", "")
        if "<decision>" not in advice or "<arguments>" not in advice:
            continue
        
        prompt = prompt_template.replace("<CONTEXT>", context).replace("<ADVICE>", advice)

        verification = client.create_completion(
            model=args.model,
            messages=[{"role": "user", "content": prompt}]
        )

        print(f"Item {item['id']}:")
        print(verification)
        print("-" * 80)

        verified_item = item.copy()
        verified_item["verification"] = verification
        verified_results.append(verified_item)

        # Save results every 5 items
        if (i + 1) % 5 == 0:
            output_filename = os.path.basename(args.results).replace(".json", "_verified.json")
            with open(os.path.join(args.output, output_filename), "w") as f:
                json.dump(verified_results, f, indent=4)

    output_filename = os.path.basename(args.results).replace(".json", "_verified.json")
    with open(os.path.join(args.output, output_filename), "w") as f:
        json.dump(verified_results, f, indent=4)
