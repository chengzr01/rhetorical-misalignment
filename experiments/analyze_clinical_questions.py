import argparse
import json

def main():
    parser = argparse.ArgumentParser(
        description='Analyze clinical decision-making questions from MIMIC-IV episodes data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=["output/small_sft_all.json", "output/small_sft_bayesian.json"],
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='analysis/small_sft_analysis.json',
        help='Path to output analysis JSON file'
    )

    args = parser.parse_args()
    
    initial_analysis = {}
    
    for input_file in args.input:
        if "bayesian" in input_file:
            with open(input_file, 'r') as f:
                data = json.load(f)
            for case in data:
                initial_analysis[case["case_id"]] = {}
                initial_analysis[case["case_id"]]["bayesian"] = {"raw_response": case["raw_principal_response"]}
    
    for input_file in args.input:
        if "all" in input_file:
            with open(input_file, 'r') as f:
                data = json.load(f)
            for case in data:
                initial_analysis[case["case_id"]][case["principal_name"]] = {"raw_response": case["raw_principal_response"]}
                
    for case_id, data in initial_analysis.items():
        for principal_type, principal_response in data.items():
            # Extract all decision, belief, and reasoning blocks from the response
            raw_response = principal_response.get("raw_response", "")
            extracted_blocks = []
            
            # Find all occurrences of decision/belief/reasoning blocks
            search_pos = 0
            while True:
                # Find next decision tag
                decision_start = raw_response.find("<decision>", search_pos)
                if decision_start == -1:
                    break
                
                decision_end = raw_response.find("</decision>", decision_start)
                if decision_end == -1:
                    break
                
                # Extract decision
                decision_text = raw_response[decision_start + len("<decision>"):decision_end].strip()
                
                # Create block dictionary
                block = {"decision": decision_text}
                
                # Look for belief tag after this decision (optional)
                belief_start = raw_response.find("<belief>", decision_end)
                belief_end = raw_response.find("</belief>", belief_start) if belief_start != -1 else -1
                if belief_start != -1 and belief_end != -1:
                    belief_text = raw_response[belief_start + len("<belief>"):belief_end].strip()
                    block["belief"] = belief_text
                
                # Look for reasoning tag after belief (or after decision if no belief)
                reasoning_search_start = belief_end if belief_end != -1 else decision_end
                reasoning_start = raw_response.find("<reasoning>", reasoning_search_start)
                reasoning_end = raw_response.find("</reasoning>", reasoning_start) if reasoning_start != -1 else -1
                reasoning_text = raw_response[reasoning_start + len("<reasoning>"):reasoning_end].strip() if reasoning_start != -1 and reasoning_end != -1 else None
                
                if reasoning_text:
                    block["reasoning"] = reasoning_text
                
                # Look for recommendation tag (optional)
                recommendation_search_start = reasoning_end if reasoning_end != -1 else (belief_end if belief_end != -1 else decision_end)
                recommendation_start = raw_response.find("<recommendation>", recommendation_search_start)
                recommendation_end = raw_response.find("</recommendation>", recommendation_start) if recommendation_start != -1 else -1
                if recommendation_start != -1 and recommendation_end != -1:
                    recommendation_text = raw_response[recommendation_start + len("<recommendation>"):recommendation_end].strip()
                    block["recommendation"] = recommendation_text
                
                extracted_blocks.append(block)
                
                # Move search position forward
                max_end = decision_end
                if belief_end != -1:
                    max_end = max(max_end, belief_end)
                if reasoning_end != -1:
                    max_end = max(max_end, reasoning_end)
                if recommendation_end != -1:
                    max_end = max(max_end, recommendation_end)
                search_pos = max_end + 1

            # Store extracted data as list of blocks
            data[principal_type]["analysis"] = extracted_blocks
    with open(args.output, 'w') as f:
        json.dump(initial_analysis, f, indent=2)

if __name__ == '__main__':
    main()