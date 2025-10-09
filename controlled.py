import os
import argparse
import json
import yaml
from tqdm import tqdm

from client import OpenRouterChatClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified", type=str, default="verified/verified.json",
                        help="Path to verified.json file containing non-exploitative arguments")
    parser.add_argument("--prompt", type=str, required=True, choices=["prompts/bayesian.yaml", "prompts/prospect.yaml"],
                        help="Simulation prompt type (bayesian or prospect)")
    parser.add_argument("--model", type=str, default="deepseek/deepseek-chat-v3.1")
    parser.add_argument("--output", type=str, default="simulations")
    args = parser.parse_args()

    # Load verified results with non-exploitative arguments
    with open(args.verified, "r") as f:
        verified_results = json.load(f)

    # Load prompt template
    with open(args.prompt, "r") as f:
        prompt_template = yaml.safe_load(f)["prompt"]

    # Initialize client
    client = OpenRouterChatClient(api_key=os.getenv("OPENROUTER_API_KEY"))

    simulations = []
    os.makedirs(args.output, exist_ok=True)

    # Process each verified result
    for item in tqdm(verified_results):
        # Extract non-exploitative arguments
        verification = item.get("verification", "")

        # Extract non-exploitative version using regex
        import re
        non_exploitative_match = re.search(r'<non-exploitative>(.*?)</non-exploitative>',
                                          verification, re.DOTALL | re.IGNORECASE)

        if not non_exploitative_match:
            print(f"Warning: No non-exploitative arguments found for item {item['id']}, skipping...")
            continue

        non_exploitative_args = non_exploitative_match.group(1).strip()

        # Extract context
        context = str(item["params"]["public"])

        # Build LLM message with non-exploitative arguments
        # Reconstruct the decision and arguments format
        original_response = item.get("response", "")
        decision_match = re.search(r'<decision>(.*?)</decision>', original_response, re.DOTALL | re.IGNORECASE)
        decision = decision_match.group(1).strip() if decision_match else "Not Guilty"

        llm_message = f"<decision>{decision}</decision>\n<arguments>\n{non_exploitative_args}\n</arguments>"

        # Build simulation prompt
        simulation_prompt = prompt_template.replace("<CONTEXT>", context).replace("<INFORMATION>", llm_message)

        # Get simulation response
        simulation_response = client.create_completion(
            model=args.model,
            messages=[{"role": "user", "content": simulation_prompt}]
        )

        print(f"\nItem {item['id']}:")
        print(f"Non-exploitative arguments: {non_exploitative_args[:100]}...")
        print(f"Simulation response: {simulation_response}")

        # Store simulation result
        simulation_item = item.copy()
        simulation_item["non_exploitative_arguments"] = non_exploitative_args
        simulation_item["controlled_llm_message"] = llm_message
        simulation_item["simulation_prompt"] = simulation_prompt
        simulation_item["simulation_response"] = simulation_response
        simulations.append(simulation_item)

    # Generate output filename
    prompt_type = "bayesian" if "bayesian" in args.prompt else "prospect"
    output_file = os.path.join(args.output, f"controlled_{prompt_type}_simulation.json")

    # Save simulations
    with open(output_file, "w") as f:
        json.dump(simulations, f, indent=4)

    print(f"\nSimulation results saved to: {output_file}")
