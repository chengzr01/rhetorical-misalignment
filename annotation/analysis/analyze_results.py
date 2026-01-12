#!/usr/bin/env python3
"""
Analyze annotation results from the USMLE Sample dataset.
Provides comprehensive statistics on belief changes, persuasion effects, and annotator behavior.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
import statistics
from datetime import datetime


# Results directory
RESULTS_DIR = '../results/usmle_sample'


def load_all_annotations():
    """Load all annotation files"""
    if not os.path.exists(RESULTS_DIR):
        return []

    annotations = []
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith('.json'):
            continue

        try:
            filepath = os.path.join(RESULTS_DIR, filename)
            with open(filepath, 'r') as f:
                annotation = json.load(f)
                annotations.append(annotation)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return annotations


def calculate_belief_change(before, after):
    """Calculate belief change, handling None values"""
    if before is None or after is None:
        return None
    return after - before


def analyze_dataset(annotations):
    """Analyze USMLE Sample annotations"""
    print("\n" + "="*80)
    print("USMLE SAMPLE DATASET ANALYSIS")
    print("="*80)

    if not annotations:
        print("No annotations found.")
        return

    print(f"\nTotal annotations: {len(annotations)}")

    # Unique cases and annotators
    unique_cases = set(a.get('case_id') for a in annotations)
    unique_annotators = set(a.get('annotator_id') for a in annotations)
    print(f"Unique cases: {len(unique_cases)}")
    print(f"Unique annotators: {len(unique_annotators)}")

    # Model breakdown
    model_counts = defaultdict(int)
    for a in annotations:
        model_key = a.get('agent_model', 'unknown')
        model_counts[model_key] += 1

    print(f"\nAnnotations by model:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count} ({count/len(annotations)*100:.1f}%)")

    # Answer correctness analysis
    print("\n" + "-"*80)
    print("ANSWER CORRECTNESS ANALYSIS")
    print("-"*80)

    step1_correct = sum(1 for a in annotations if a.get('step1', {}).get('is_correct'))
    step2_correct = sum(1 for a in annotations if a.get('step2', {}).get('is_correct'))
    step3_correct = sum(1 for a in annotations if a.get('step3', {}).get('is_correct'))

    print(f"\nStep 1 (Initial answer): {step1_correct}/{len(annotations)} correct ({step1_correct/len(annotations)*100:.1f}%)")
    print(f"Step 2 (After AI): {step2_correct}/{len(annotations)} correct ({step2_correct/len(annotations)*100:.1f}%)")
    print(f"Step 3 (After truth): {step3_correct}/{len(annotations)} correct ({step3_correct/len(annotations)*100:.1f}%)")

    # Answer change analysis
    print("\n" + "-"*80)
    print("ANSWER CHANGE ANALYSIS")
    print("-"*80)

    changed_1_to_2 = sum(1 for a in annotations if a.get('step1_to_step2_changes', {}).get('answer_changed'))
    changed_2_to_3 = sum(1 for a in annotations if a.get('step2_to_step3_changes', {}).get('answer_changed'))

    print(f"\nStep 1 → Step 2 (After seeing AI): {changed_1_to_2}/{len(annotations)} changed answer ({changed_1_to_2/len(annotations)*100:.1f}%)")
    print(f"Step 2 → Step 3 (After seeing truth): {changed_2_to_3}/{len(annotations)} changed answer ({changed_2_to_3/len(annotations)*100:.1f}%)")

    # Belief change analysis
    print("\n" + "-"*80)
    print("BELIEF CHANGE ANALYSIS")
    print("-"*80)

    belief_changes_1_2 = []
    belief_changes_2_3 = []

    for annotation in annotations:
        step1 = annotation.get('step1', {})
        step2 = annotation.get('step2', {})
        step3 = annotation.get('step3', {})

        belief1 = step1.get('answer_belief')
        belief2 = step2.get('answer_belief')
        belief3 = step3.get('answer_belief')

        change_1_2 = calculate_belief_change(belief1, belief2)
        change_2_3 = calculate_belief_change(belief2, belief3)

        if change_1_2 is not None:
            belief_changes_1_2.append(change_1_2)
        if change_2_3 is not None:
            belief_changes_2_3.append(change_2_3)

    # Step 1 to 2
    if belief_changes_1_2:
        increased_1_2 = [x for x in belief_changes_1_2 if x > 0]
        decreased_1_2 = [x for x in belief_changes_1_2 if x < 0]
        unchanged_1_2 = [x for x in belief_changes_1_2 if x == 0]

        print(f"\nStep 1 → Step 2 (After seeing AI response):")
        print(f"  Belief increased: {len(increased_1_2)}/{len(belief_changes_1_2)} ({len(increased_1_2)/len(belief_changes_1_2)*100:.1f}%)")
        if increased_1_2:
            print(f"    Mean increase: {statistics.mean(increased_1_2):.3f}")
            print(f"    Median increase: {statistics.median(increased_1_2):.3f}")
            print(f"    Min: {min(increased_1_2):.3f}, Max: {max(increased_1_2):.3f}")
            print(f"    Std dev: {statistics.stdev(increased_1_2):.3f}")

        print(f"  Belief decreased: {len(decreased_1_2)}/{len(belief_changes_1_2)} ({len(decreased_1_2)/len(belief_changes_1_2)*100:.1f}%)")
        if decreased_1_2:
            print(f"    Mean decrease: {statistics.mean(decreased_1_2):.3f}")
            print(f"    Median decrease: {statistics.median(decreased_1_2):.3f}")
            print(f"    Min: {min(decreased_1_2):.3f}, Max: {max(decreased_1_2):.3f}")
            print(f"    Std dev: {statistics.stdev(decreased_1_2):.3f}")

        print(f"  Belief unchanged: {len(unchanged_1_2)}/{len(belief_changes_1_2)} ({len(unchanged_1_2)/len(belief_changes_1_2)*100:.1f}%)")

    # Step 2 to 3
    if belief_changes_2_3:
        increased_2_3 = [x for x in belief_changes_2_3 if x > 0]
        decreased_2_3 = [x for x in belief_changes_2_3 if x < 0]
        unchanged_2_3 = [x for x in belief_changes_2_3 if x == 0]

        print(f"\nStep 2 → Step 3 (After seeing ground truth):")
        print(f"  Belief increased: {len(increased_2_3)}/{len(belief_changes_2_3)} ({len(increased_2_3)/len(belief_changes_2_3)*100:.1f}%)")
        if increased_2_3:
            print(f"    Mean increase: {statistics.mean(increased_2_3):.3f}")
            print(f"    Median increase: {statistics.median(increased_2_3):.3f}")
            print(f"    Min: {min(increased_2_3):.3f}, Max: {max(increased_2_3):.3f}")
            print(f"    Std dev: {statistics.stdev(increased_2_3):.3f}")

        print(f"  Belief decreased: {len(decreased_2_3)}/{len(belief_changes_2_3)} ({len(decreased_2_3)/len(belief_changes_2_3)*100:.1f}%)")
        if decreased_2_3:
            print(f"    Mean decrease: {statistics.mean(decreased_2_3):.3f}")
            print(f"    Median decrease: {statistics.median(decreased_2_3):.3f}")
            print(f"    Min: {min(decreased_2_3):.3f}, Max: {max(decreased_2_3):.3f}")
            print(f"    Std dev: {statistics.stdev(decreased_2_3):.3f}")

        print(f"  Belief unchanged: {len(unchanged_2_3)}/{len(belief_changes_2_3)} ({len(unchanged_2_3)/len(belief_changes_2_3)*100:.1f}%)")

    # Persuasion analysis: Did AI persuade annotators to wrong answer?
    print("\n" + "-"*80)
    print("PERSUASION ANALYSIS")
    print("-"*80)

    correct_to_incorrect = 0
    incorrect_to_correct = 0
    stayed_correct = 0
    stayed_incorrect = 0

    # Track with belief changes
    c_to_i_belief_changes = []
    i_to_c_belief_changes = []

    for annotation in annotations:
        step1 = annotation.get('step1', {})
        step2 = annotation.get('step2', {})

        step1_correct = step1.get('is_correct', False)
        step2_correct = step2.get('is_correct', False)
        changed = annotation.get('step1_to_step2_changes', {}).get('answer_changed', False)

        belief_change = calculate_belief_change(step1.get('answer_belief'), step2.get('answer_belief'))

        if changed:
            if step1_correct and not step2_correct:
                correct_to_incorrect += 1
                if belief_change is not None:
                    c_to_i_belief_changes.append(belief_change)
            elif not step1_correct and step2_correct:
                incorrect_to_correct += 1
                if belief_change is not None:
                    i_to_c_belief_changes.append(belief_change)
        else:
            if step1_correct:
                stayed_correct += 1
            else:
                stayed_incorrect += 1

    print(f"\nFrom Step 1 to Step 2 (AI persuasion):")
    print(f"  Persuaded from correct → incorrect: {correct_to_incorrect}/{len(annotations)} ({correct_to_incorrect/len(annotations)*100:.1f}%)")
    if c_to_i_belief_changes:
        print(f"    Mean belief change: {statistics.mean(c_to_i_belief_changes):.3f}")
        print(f"    Median belief change: {statistics.median(c_to_i_belief_changes):.3f}")

    print(f"  Persuaded from incorrect → correct: {incorrect_to_correct}/{len(annotations)} ({incorrect_to_correct/len(annotations)*100:.1f}%)")
    if i_to_c_belief_changes:
        print(f"    Mean belief change: {statistics.mean(i_to_c_belief_changes):.3f}")
        print(f"    Median belief change: {statistics.median(i_to_c_belief_changes):.3f}")

    print(f"  Stayed correct: {stayed_correct}/{len(annotations)} ({stayed_correct/len(annotations)*100:.1f}%)")
    print(f"  Stayed incorrect: {stayed_incorrect}/{len(annotations)} ({stayed_incorrect/len(annotations)*100:.1f}%)")

    # Highlights analysis
    print("\n" + "-"*80)
    print("HIGHLIGHTS ANALYSIS")
    print("-"*80)

    total_highlights = 0
    highlights_by_step = defaultdict(int)

    for annotation in annotations:
        highlights = annotation.get('highlights', [])
        total_highlights += len(highlights)
        for highlight in highlights:
            step = highlight.get('step', 'unknown')
            highlights_by_step[step] += 1

    print(f"\nTotal highlights: {total_highlights}")
    annotations_with_highlights = sum(1 for a in annotations if a.get('highlights'))
    print(f"Annotations with highlights: {annotations_with_highlights}/{len(annotations)} ({annotations_with_highlights/len(annotations)*100:.1f}%)")

    if highlights_by_step:
        print(f"\nHighlights by step:")
        for step, count in sorted(highlights_by_step.items()):
            print(f"  {step}: {count}")

    # Time analysis
    print("\n" + "-"*80)
    print("TIME ANALYSIS")
    print("-"*80)

    time_step1 = []
    time_step2 = []
    time_step3 = []

    for annotation in annotations:
        try:
            start = datetime.fromisoformat(annotation.get('session_start'))
            step1_time = datetime.fromisoformat(annotation.get('step1_time'))
            step2_time = datetime.fromisoformat(annotation.get('step2_time'))
            step3_time = datetime.fromisoformat(annotation.get('step3_time'))

            time_step1.append((step1_time - start).total_seconds())
            time_step2.append((step2_time - step1_time).total_seconds())
            time_step3.append((step3_time - step2_time).total_seconds())
        except:
            pass

    if time_step1:
        print(f"\nTime spent on Step 1 (seconds):")
        print(f"  Mean: {statistics.mean(time_step1):.1f}")
        print(f"  Median: {statistics.median(time_step1):.1f}")
        print(f"  Min: {min(time_step1):.1f}, Max: {max(time_step1):.1f}")

    if time_step2:
        print(f"\nTime spent on Step 2 (seconds):")
        print(f"  Mean: {statistics.mean(time_step2):.1f}")
        print(f"  Median: {statistics.median(time_step2):.1f}")
        print(f"  Min: {min(time_step2):.1f}, Max: {max(time_step2):.1f}")

    if time_step3:
        print(f"\nTime spent on Step 3 (seconds):")
        print(f"  Mean: {statistics.mean(time_step3):.1f}")
        print(f"  Median: {statistics.median(time_step3):.1f}")
        print(f"  Min: {min(time_step3):.1f}, Max: {max(time_step3):.1f}")

    # Annotator demographics
    print("\n" + "-"*80)
    print("ANNOTATOR DEMOGRAPHICS")
    print("-"*80)

    demographics_data = defaultdict(lambda: defaultdict(int))
    for annotation in annotations:
        demo = annotation.get('demographics', {})
        for key, value in demo.items():
            if key != 'submitted_at' and value and str(value).strip():
                demographics_data[key][str(value)] += 1

    if demographics_data:
        for key, values in sorted(demographics_data.items()):
            print(f"\n{key.replace('_', ' ').title()}:")
            for value, count in sorted(values.items(), key=lambda x: x[1], reverse=True):
                print(f"  {value}: {count} ({count/len(annotations)*100:.1f}%)")
    else:
        print("\nNo demographic information available (all anonymous).")


def main():
    """Main analysis function"""
    print("\n" + "="*80)
    print("ANNOTATION RESULTS ANALYSIS")
    print("="*80)

    annotations = load_all_annotations()
    if annotations:
        analyze_dataset(annotations)
    else:
        print("\nNo annotations found in", RESULTS_DIR)

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
