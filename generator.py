import os
import argparse
import json
import yaml
from tqdm import tqdm

from client import OpenRouterChatClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="datasets/old_bailey.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--prompt", type=str, default="prompts/law.yaml")
    parser.add_argument("--model", type=str, default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()

    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    with open(args.prompt, "r") as f:
        prompt_template = yaml.safe_load(f)["prompt"]

    client = OpenRouterChatClient(api_key=os.getenv("OPENROUTER_API_KEY"))

    results = []
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for i, item in enumerate(tqdm(dataset[:args.limit])):
        context = str(item["params"]["public"])
        information = str(item["params"]["private"])

        prompt = prompt_template.replace("<CONTEXT>", context).replace("<INFORMATION>", information)

        response = client.create_completion(
            model=args.model,
            messages=[{"role": "user", "content": prompt}]
        )

        print(f"Item {item['id']}:")
        print(response)

        result_item = item.copy()
        result_item["response"] = response
        results.append(result_item)

        # Save results every 10 items
        if (i + 1) % 10 == 0:
            with open(args.results + f"/{args.model}_{args.limit}.json", "w") as f:
                json.dump(results, f, indent=4)

    with open(args.results + f"/{args.model}_{args.limit}.json", "w") as f:
        json.dump(results, f, indent=4)