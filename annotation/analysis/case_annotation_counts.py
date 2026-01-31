#!/usr/bin/env python3
"""
Generate a comprehensive report on case annotation analysis:
1. Original cases each model has
2. Annotated cases each model has
3. Distribution of cases across different steps (Step1, Step2, Step3)
4. Distribution of harmful/helpful persuasion across cases
"""

import json
import os
from collections import defaultdict

# Directories
RESULTS_DIR = '../results/usmle_sample'
DATASETS_DIR = '../datasets/usmle_sample'
PERSUASION_FILE = 'persuasion_examples.json'

# Model mapping to identify which model each annotation belongs to
MODEL_PATTERNS = {
    'llama_small': ['llama-3.1-8b-instruct'],
    'llama': ['llama-3.3', '70b-instruct'],
    'llama_large': ['405b', 'llama-3.1-405b'],
    'llama_dpo': ['tulu-3-8b-dpo', '8b-dpo'],
    'llama_sft': ['tulu-3-8b-sft', '8b-sft'],
    'deepseek': ['deepseek'],
}


def identify_model(agent_model_str):
    """Identify which model an agent_model string belongs to"""
    if not agent_model_str:
        return 'unknown'

    agent_model_lower = agent_model_str.lower()

    for model_key, patterns in MODEL_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in agent_model_lower:
                return model_key

    return 'unknown'


def load_original_cases():
    """Load all original cases from the dataset files"""
    cases = []

    if not os.path.exists(DATASETS_DIR):
        print(f"Warning: Dataset directory {DATASETS_DIR} not found")
        return cases

    # Load all step files
    for step_file in ['Step1_questions_parsed.json', 'Step2_CK_questions_parsed.json', 'Step3_questions_parsed.json']:
        filepath = os.path.join(DATASETS_DIR, step_file)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    step_cases = json.load(f)
                    step_name = step_file.split('_')[0]  # Extract 'Step1', 'Step2', 'Step3'
                    for idx, case in enumerate(step_cases):
                        cases.append({
                            'case_id': f"usmle_sample_{idx}",
                            'step': step_name,
                            'exam': case.get('exam', step_name)
                        })
            except Exception as e:
                print(f"Error loading {step_file}: {e}")

    return cases


def load_all_annotations():
    """Load all annotation files and organize by case_id and model"""
    if not os.path.exists(RESULTS_DIR):
        return {}, {}

    # Structure: {case_id: {model_key: [annotation1, annotation2, ...]}}
    annotations_by_case_and_model = defaultdict(lambda: defaultdict(list))
    # Structure: {model_key: set of case_ids}
    cases_by_model = defaultdict(set)

    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith('.json'):
            continue

        try:
            filepath = os.path.join(RESULTS_DIR, filename)
            with open(filepath, 'r') as f:
                annotation = json.load(f)
                case_id = annotation.get('case_id')
                agent_model = annotation.get('agent_model', '')

                if case_id:
                    model_key = identify_model(agent_model)
                    annotations_by_case_and_model[case_id][model_key].append(annotation)
                    cases_by_model[model_key].add(case_id)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return annotations_by_case_and_model, cases_by_model


def load_persuasion_data():
    """Load persuasion examples data"""
    if not os.path.exists(PERSUASION_FILE):
        print(f"Warning: {PERSUASION_FILE} not found")
        return None

    try:
        with open(PERSUASION_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading persuasion data: {e}")
        return None


def extract_step_from_case_id(case_id):
    """Extract step information from case_id by looking at the number"""
    # usmle_sample_0 to usmle_sample_117 maps to Step1 (0-117)
    # usmle_sample_118 to usmle_sample_251 maps to Step2 (118-251)
    # usmle_sample_252+ maps to Step3
    try:
        case_num = int(case_id.split('_')[-1])
        if case_num <= 117:
            return 'Step1'
        elif case_num <= 251:
            return 'Step2'
        else:
            return 'Step3'
    except:
        return 'Unknown'


def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*100)
    print(title)
    print("="*100)


def analyze_original_and_annotated_cases():
    """Analyze original cases vs annotated cases per model"""
    print_section_header("SECTION 1: ORIGINAL CASES vs ANNOTATED CASES BY MODEL")

    annotations_data, cases_by_model = load_all_annotations()

    if not annotations_data:
        print("\nNo annotations found.")
        return annotations_data, cases_by_model

    # Get all models
    all_models = sorted(cases_by_model.keys())

    print(f"\nModels analyzed: {', '.join(all_models)}\n")

    # For each model, show original cases (cases assigned) vs annotated cases
    print(f"{'Model':<20} {'Original Cases':<20} {'Annotated Cases':<20} {'Completion Rate':<20}")
    print("-"*80)

    for model in all_models:
        original_count = len(cases_by_model[model])
        annotated_cases = {cid for cid in cases_by_model[model]
                          if len(annotations_data.get(cid, {}).get(model, [])) > 0}
        annotated_count = len(annotated_cases)
        completion_rate = (annotated_count / original_count * 100) if original_count > 0 else 0

        print(f"{model:<20} {original_count:<20} {annotated_count:<20} {completion_rate:.1f}%")

    return annotations_data, cases_by_model


def analyze_step_distribution(annotations_data, cases_by_model):
    """Analyze distribution of cases across different steps"""
    print_section_header("SECTION 2: DISTRIBUTION OF CASES ACROSS STEPS")

    all_models = sorted(cases_by_model.keys())

    # Count cases by step for each model
    step_distribution = defaultdict(lambda: defaultdict(int))

    for model in all_models:
        for case_id in cases_by_model[model]:
            step = extract_step_from_case_id(case_id)
            step_distribution[model][step] += 1

    # Print distribution table
    steps = ['Step1', 'Step2', 'Step3']
    print(f"\n{'Model':<20} {'Step1':<15} {'Step2':<15} {'Step3':<15} {'Total':<15}")
    print("-"*80)

    for model in all_models:
        step_counts = step_distribution[model]
        total = sum(step_counts.values())
        row = f"{model:<20}"
        for step in steps:
            count = step_counts[step]
            pct = (count / total * 100) if total > 0 else 0
            row += f"{count} ({pct:.1f}%){' '*(15-len(str(count))-8)}"
        row += f"{total:<15}"
        print(row)

    # Overall distribution
    print("-"*80)
    overall_counts = defaultdict(int)
    for model in all_models:
        for step, count in step_distribution[model].items():
            overall_counts[step] += count

    total_overall = sum(overall_counts.values())
    row = f"{'TOTAL':<20}"
    for step in steps:
        count = overall_counts[step]
        pct = (count / total_overall * 100) if total_overall > 0 else 0
        row += f"{count} ({pct:.1f}%){' '*(15-len(str(count))-8)}"
    row += f"{total_overall:<15}"
    print(row)


def analyze_persuasion_distribution():
    """Analyze distribution of harmful/helpful persuasion across cases"""
    print_section_header("SECTION 3: PERSUASION DISTRIBUTION (Harmful vs Helpful)")

    persuasion_data = load_persuasion_data()

    if not persuasion_data:
        print("\nNo persuasion data available.")
        return

    # Print summary
    summary = persuasion_data.get('summary', {})
    harmful_count = summary.get('harmful_persuasion', {}).get('count', 0)
    helpful_count = summary.get('helpful_persuasion', {}).get('count', 0)
    total_persuasion = harmful_count + helpful_count

    print(f"\nTotal persuasion cases: {total_persuasion}")
    print(f"  Harmful persuasion (correct → incorrect): {harmful_count} ({harmful_count/total_persuasion*100:.1f}%)")
    print(f"  Helpful persuasion (incorrect → correct): {helpful_count} ({helpful_count/total_persuasion*100:.1f}%)")
    print(f"  Net helpful persuasions: {helpful_count - harmful_count}")

    # Analyze by model
    print("\n" + "-"*100)
    print("PERSUASION BY MODEL")
    print("-"*100)

    model_persuasion = defaultdict(lambda: {'harmful': 0, 'helpful': 0})

    cases = persuasion_data.get('cases', {})
    for persuasion_type in ['harmful_persuasion', 'helpful_persuasion']:
        for case in cases.get(persuasion_type, []):
            model = identify_model(case.get('model', ''))
            if persuasion_type == 'harmful_persuasion':
                model_persuasion[model]['harmful'] += 1
            else:
                model_persuasion[model]['helpful'] += 1

    print(f"\n{'Model':<20} {'Harmful':<15} {'Helpful':<15} {'Total':<15} {'Net Helpful':<15}")
    print("-"*80)

    for model in sorted(model_persuasion.keys()):
        harmful = model_persuasion[model]['harmful']
        helpful = model_persuasion[model]['helpful']
        total = harmful + helpful
        net = helpful - harmful
        print(f"{model:<20} {harmful:<15} {helpful:<15} {total:<15} {net:<15}")

    # Analyze by step
    print("\n" + "-"*100)
    print("PERSUASION BY STEP")
    print("-"*100)

    step_persuasion = defaultdict(lambda: {'harmful': 0, 'helpful': 0})

    for persuasion_type in ['harmful_persuasion', 'helpful_persuasion']:
        for case in cases.get(persuasion_type, []):
            case_id = case.get('case_id', '')
            step = extract_step_from_case_id(case_id)
            if persuasion_type == 'harmful_persuasion':
                step_persuasion[step]['harmful'] += 1
            else:
                step_persuasion[step]['helpful'] += 1

    print(f"\n{'Step':<20} {'Harmful':<15} {'Helpful':<15} {'Total':<15} {'Net Helpful':<15}")
    print("-"*80)

    for step in ['Step1', 'Step2', 'Step3']:
        harmful = step_persuasion[step]['harmful']
        helpful = step_persuasion[step]['helpful']
        total = harmful + helpful
        net = helpful - harmful
        if total > 0:  # Only show steps with data
            print(f"{step:<20} {harmful:<15} {helpful:<15} {total:<15} {net:<15}")


def analyze_case_overlap(cases_by_model):
    """Analyze overlap of cases across different models"""
    print_section_header("SECTION 4: CASE OVERLAP ACROSS MODELS")

    all_models = sorted(cases_by_model.keys())

    print("\nPairwise case overlap between models:\n")

    # Create a matrix of overlaps
    print(f"{'Model 1':<20} {'Model 2':<20} {'Overlap':<15} {'Unique to M1':<15} {'Unique to M2':<15}")
    print("-"*85)

    for i, model1 in enumerate(all_models):
        for model2 in all_models[i+1:]:
            cases1 = cases_by_model[model1]
            cases2 = cases_by_model[model2]

            overlap = cases1 & cases2
            unique_to_m1 = cases1 - cases2
            unique_to_m2 = cases2 - cases1

            print(f"{model1:<20} {model2:<20} {len(overlap):<15} {len(unique_to_m1):<15} {len(unique_to_m2):<15}")

    # Special focus on llama_small, llama_dpo, llama_sft
    print("\n" + "-"*100)
    print("DETAILED ANALYSIS: llama_small, llama_dpo, llama_sft")
    print("-"*100)

    target_models = ['llama_small', 'llama_dpo', 'llama_sft']

    # Check if all target models exist
    overlap_categories = {}
    if all(m in cases_by_model for m in target_models):
        small_cases = cases_by_model['llama_small']
        dpo_cases = cases_by_model['llama_dpo']
        sft_cases = cases_by_model['llama_sft']

        # Three-way analysis
        all_three = small_cases & dpo_cases & sft_cases
        small_and_dpo = (small_cases & dpo_cases) - sft_cases
        small_and_sft = (small_cases & sft_cases) - dpo_cases
        dpo_and_sft = (dpo_cases & sft_cases) - small_cases
        only_small = small_cases - dpo_cases - sft_cases
        only_dpo = dpo_cases - small_cases - sft_cases
        only_sft = sft_cases - small_cases - dpo_cases

        # Store for later persuasion analysis
        overlap_categories = {
            'all_three': all_three,
            'small_and_dpo': small_and_dpo,
            'small_and_sft': small_and_sft,
            'dpo_and_sft': dpo_and_sft,
            'only_small': only_small,
            'only_dpo': only_dpo,
            'only_sft': only_sft
        }

        print(f"\nTotal cases per model:")
        print(f"  llama_small: {len(small_cases)}")
        print(f"  llama_dpo:   {len(dpo_cases)}")
        print(f"  llama_sft:   {len(sft_cases)}")

        print(f"\nCase distribution:")
        print(f"  Common to all three models:          {len(all_three):3d} cases")
        print(f"  Common to small & dpo (not sft):     {len(small_and_dpo):3d} cases")
        print(f"  Common to small & sft (not dpo):     {len(small_and_sft):3d} cases")
        print(f"  Common to dpo & sft (not small):     {len(dpo_and_sft):3d} cases")
        print(f"  Unique to llama_small:               {len(only_small):3d} cases")
        print(f"  Unique to llama_dpo:                 {len(only_dpo):3d} cases")
        print(f"  Unique to llama_sft:                 {len(only_sft):3d} cases")

        total_unique_cases = len(small_cases | dpo_cases | sft_cases)
        print(f"\nTotal unique cases across these 3 models: {total_unique_cases}")

        # Show some examples of unique cases for each model
        print("\n" + "-"*100)
        print("Sample of unique cases (first 10 for each):")
        print("-"*100)

        if only_small:
            print(f"\nUnique to llama_small ({len(only_small)} total):")
            for case in sorted(only_small, key=lambda x: int(x.split('_')[-1]))[:10]:
                step = extract_step_from_case_id(case)
                print(f"  {case} ({step})")

        if only_dpo:
            print(f"\nUnique to llama_dpo ({len(only_dpo)} total):")
            for case in sorted(only_dpo, key=lambda x: int(x.split('_')[-1]))[:10]:
                step = extract_step_from_case_id(case)
                print(f"  {case} ({step})")

        if only_sft:
            print(f"\nUnique to llama_sft ({len(only_sft)} total):")
            for case in sorted(only_sft, key=lambda x: int(x.split('_')[-1]))[:10]:
                step = extract_step_from_case_id(case)
                print(f"  {case} ({step})")

    # Overall statistics
    print("\n" + "-"*100)
    print("OVERALL CASE DISTRIBUTION")
    print("-"*100)

    all_cases = set()
    for cases in cases_by_model.values():
        all_cases.update(cases)

    print(f"\nTotal unique cases across all models: {len(all_cases)}")

    # Count how many models each case appears in
    case_model_count = defaultdict(int)
    for case in all_cases:
        for model_cases in cases_by_model.values():
            if case in model_cases:
                case_model_count[case] += 1

    model_count_distribution = defaultdict(int)
    for count in case_model_count.values():
        model_count_distribution[count] += 1

    print(f"\nDistribution of cases by number of models:")
    for count in sorted(model_count_distribution.keys()):
        num_cases = model_count_distribution[count]
        pct = (num_cases / len(all_cases) * 100) if len(all_cases) > 0 else 0
        print(f"  Cases in {count} model(s): {num_cases:3d} ({pct:.1f}%)")

    return overlap_categories


def analyze_persuasion_by_overlap(overlap_categories):
    """Analyze harmful/helpful persuasion for each case overlap category"""
    print("\n" + "="*100)
    print("PERSUASION ANALYSIS BY CASE OVERLAP CATEGORY (BY MODEL)")
    print("="*100)

    if not overlap_categories:
        print("\nNo overlap categories available.")
        return

    persuasion_data = load_persuasion_data()
    if not persuasion_data:
        print("\nNo persuasion data available.")
        return

    # Build a mapping of case_id + model -> persuasion type
    persuasion_map = {}
    cases = persuasion_data.get('cases', {})

    for persuasion_type in ['harmful_persuasion', 'helpful_persuasion']:
        for case in cases.get(persuasion_type, []):
            case_id = case.get('case_id', '')
            model = identify_model(case.get('model', ''))
            key = (case_id, model)
            persuasion_map[key] = persuasion_type

    # Analyze persuasion for each overlap category
    category_labels = {
        'all_three': ('Common to all three (small, dpo, sft)', ['llama_small', 'llama_dpo', 'llama_sft']),
        'small_and_dpo': ('Common to small & dpo (not sft)', ['llama_small', 'llama_dpo']),
        'small_and_sft': ('Common to small & sft (not dpo)', ['llama_small', 'llama_sft']),
        'dpo_and_sft': ('Common to dpo & sft (not small)', ['llama_dpo', 'llama_sft']),
        'only_small': ('Unique to llama_small', ['llama_small']),
        'only_dpo': ('Unique to llama_dpo', ['llama_dpo']),
        'only_sft': ('Unique to llama_sft', ['llama_sft'])
    }

    for category_key, (label, models_to_check) in category_labels.items():
        if category_key not in overlap_categories:
            continue

        case_set = overlap_categories[category_key]

        print(f"\n{label}:")
        print(f"  Total cases in this category: {len(case_set)}")
        print(f"  Models involved: {', '.join(models_to_check)}")
        print()

        # Count persuasion by model
        model_stats = {}
        for model in models_to_check:
            harmful_count = 0
            helpful_count = 0

            for case_id in case_set:
                key = (case_id, model)
                if key in persuasion_map:
                    if persuasion_map[key] == 'harmful_persuasion':
                        harmful_count += 1
                    else:
                        helpful_count += 1

            model_stats[model] = {
                'harmful': harmful_count,
                'helpful': helpful_count,
                'total': harmful_count + helpful_count
            }

        # Print table
        print(f"  {'Model':<20} {'Harmful':<12} {'Helpful':<12} {'Total':<12} {'Net':<12}")
        print(f"  {'-'*68}")

        for model in models_to_check:
            stats = model_stats[model]
            harmful = stats['harmful']
            helpful = stats['helpful']
            total = stats['total']
            net = helpful - harmful
            net_str = f"+{net}" if net >= 0 else str(net)

            print(f"  {model:<20} {harmful:<12} {helpful:<12} {total:<12} {net_str:<12}")

        # Overall for category
        total_harmful = sum(model_stats[m]['harmful'] for m in models_to_check)
        total_helpful = sum(model_stats[m]['helpful'] for m in models_to_check)
        total_persuasion = total_harmful + total_helpful
        overall_net = total_helpful - total_harmful
        overall_net_str = f"+{overall_net}" if overall_net >= 0 else str(overall_net)

        print(f"  {'-'*68}")
        print(f"  {'TOTAL':<20} {total_harmful:<12} {total_helpful:<12} {total_persuasion:<12} {overall_net_str:<12}")

        # Calculate persuasion rate
        persuasion_rate = (total_persuasion / (len(case_set) * len(models_to_check)) * 100) if len(case_set) > 0 else 0
        print(f"  Persuasion rate: {persuasion_rate:.1f}% ({total_persuasion} persuasions / {len(case_set)} cases × {len(models_to_check)} models)")

    # Detailed breakdown with case examples
    print("\n" + "-"*100)
    print("DETAILED EXAMPLES BY CATEGORY")
    print("-"*100)

    for category_key, (label, models_to_check) in category_labels.items():
        if category_key not in overlap_categories:
            continue

        case_set = overlap_categories[category_key]

        # Collect persuasion examples by model
        model_examples = {model: {'harmful': [], 'helpful': []} for model in models_to_check}

        for case_id in sorted(case_set, key=lambda x: int(x.split('_')[-1])):
            for model in models_to_check:
                key = (case_id, model)
                if key in persuasion_map:
                    if persuasion_map[key] == 'harmful_persuasion':
                        model_examples[model]['harmful'].append(case_id)
                    else:
                        model_examples[model]['helpful'].append(case_id)

        # Check if there are any persuasions in this category
        has_persuasion = any(
            len(model_examples[m]['harmful']) + len(model_examples[m]['helpful']) > 0
            for m in models_to_check
        )

        if has_persuasion:
            print(f"\n{label}:")
            for model in models_to_check:
                harmful_cases = model_examples[model]['harmful']
                helpful_cases = model_examples[model]['helpful']

                if harmful_cases or helpful_cases:
                    print(f"\n  {model}:")

                if harmful_cases:
                    print(f"    Harmful ({len(harmful_cases)}): ", end="")
                    display_cases = harmful_cases[:5]
                    print(", ".join(display_cases), end="")
                    if len(harmful_cases) > 5:
                        print(f", ... (+{len(harmful_cases)-5} more)")
                    else:
                        print()

                if helpful_cases:
                    print(f"    Helpful ({len(helpful_cases)}): ", end="")
                    display_cases = helpful_cases[:5]
                    print(", ".join(display_cases), end="")
                    if len(helpful_cases) > 5:
                        print(f", ... (+{len(helpful_cases)-5} more)")
                    else:
                        print()


def analyze_annotation_details(annotations_data, cases_by_model):
    """Provide detailed annotation statistics"""
    print_section_header("SECTION 5: DETAILED ANNOTATION STATISTICS")

    if not annotations_data:
        print("\nNo annotations found.")
        return

    # Get all case IDs sorted
    all_case_ids = sorted(annotations_data.keys(),
                         key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)

    # Get all models
    all_models = sorted(cases_by_model.keys())

    print(f"\nTotal unique cases with annotations: {len(all_case_ids)}")
    print(f"Total annotation files: {sum(len(annotations_data[cid][m]) for cid in all_case_ids for m in all_models)}")

    # Statistics
    total_annotations = 0
    model_totals = defaultdict(int)

    for case_id in all_case_ids:
        case_data = annotations_data[case_id]
        for model in all_models:
            count = len(case_data.get(model, []))
            model_totals[model] += count
            total_annotations += count

    print(f"\nPer-model annotation counts:")
    for model in all_models:
        count = model_totals[model]
        case_count = len(cases_by_model[model])
        avg = count / case_count if case_count > 0 else 0
        print(f"  {model:15s}: {count:4d} annotations across {case_count:3d} cases (avg {avg:.2f} per case)")


def main():
    print("\n" + "="*100)
    print("COMPREHENSIVE CASE ANNOTATION ANALYSIS REPORT")
    print("Dataset: USMLE Sample")
    print("="*100)

    # Section 1: Original vs Annotated Cases
    annotations_data, cases_by_model = analyze_original_and_annotated_cases()

    # Section 2: Step Distribution
    if annotations_data:
        analyze_step_distribution(annotations_data, cases_by_model)

    # Section 3: Persuasion Distribution
    analyze_persuasion_distribution()

    # Section 4: Case Overlap Analysis
    overlap_categories = {}
    if cases_by_model:
        overlap_categories = analyze_case_overlap(cases_by_model)

    # Section 4b: Persuasion by Overlap Category
    if overlap_categories:
        analyze_persuasion_by_overlap(overlap_categories)

    # Section 5: Detailed Statistics
    analyze_annotation_details(annotations_data, cases_by_model)

    print("\n" + "="*100)
    print("END OF REPORT")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
