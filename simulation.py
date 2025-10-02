import os
import argparse
import json
import yaml
from tqdm import tqdm

from client import OpenRouterChatClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--prompt", type=str, required=True, choices=["prompts/bayesian.yaml", "prompts/prospect.yaml"],
                        help="Simulation prompt type (bayesian or prospect)")
    parser.add_argument("--model", type=str, default="deepseek/deepseek-chat-v3.1")
    parser.add_argument("--output", type=str, default="simulations")
    args = parser.parse_args()

    # Load results
    with open(args.results, "r") as f:
        results = json.load(f)

    # Load prompt template
    with open(args.prompt, "r") as f:
        prompt_template = yaml.safe_load(f)["prompt"]

    # Initialize client
    client = OpenRouterChatClient(api_key=os.getenv("OPENROUTER_API_KEY"))

    simulations = []
    os.makedirs(args.output, exist_ok=True)

    # Process each result
    for item in tqdm(results):
        # Skip items without a response
        if not item.get("response"):
            continue

        # Extract context and LLM-generated message
        context = str(item["params"]["public"])
        llm_message = item["response"]

        # Build simulation prompt
        simulation_prompt = prompt_template.replace("<CONTEXT>", context).replace("<INFORMATION>", llm_message)

        # Get simulation response
        simulation_response = client.create_completion(
            model=args.model,
            messages=[{"role": "user", "content": simulation_prompt}]
        )

        print(f"\nItem {item['id']}:")
        print(f"Simulation response: {simulation_response}")

        # Store simulation result
        simulation_item = item.copy()
        simulation_item["simulation_prompt"] = simulation_prompt
        simulation_item["simulation_response"] = simulation_response
        simulations.append(simulation_item)

    # Generate output filename
    results_basename = os.path.basename(args.results).replace(".json", "")
    prompt_type = "bayesian" if "bayesian" in args.prompt else "prospect"
    output_file = os.path.join(args.output, f"{results_basename}_{prompt_type}_simulation.json")

    # Save simulations
    with open(output_file, "w") as f:
        json.dump(simulations, f, indent=4)

    print(f"\nSimulation results saved to: {output_file}")
