#!/usr/bin/env python3
"""
Compare persuasion effectiveness across different AI models for USMLE Sample dataset.
Shows which models are more successful at changing annotator beliefs.
"""

import json
import os
from collections import defaultdict
import statistics


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


def compare_models(annotations):
    """Compare models for USMLE dataset"""
    print("\n" + "="*80)
    print("USMLE SAMPLE - MODEL COMPARISON")
    print("="*80)

    if not annotations:
        print("No annotations found.")
        return

    # Group annotations by model
    by_model = defaultdict(list)
    for annotation in annotations:
        model_key = annotation.get('agent_model', 'unknown')
        by_model[model_key].append(annotation)

    print(f"\nTotal annotations: {len(annotations)}")
    print(f"Models found: {len(by_model)} unique models")

    # Analyze each model
    model_stats = {}

    for model_key, model_annotations in sorted(by_model.items()):
        stats = {
            'n': len(model_annotations),
            'answer_changed': 0,
            'belief_changes': [],
            'correct_to_incorrect': 0,
            'incorrect_to_correct': 0,
            'step1_correct': 0,
            'step2_correct': 0,
            'step3_correct': 0
        }

        for annotation in model_annotations:
            step1 = annotation.get('step1', {})
            step2 = annotation.get('step2', {})
            step3 = annotation.get('step3', {})
            changes_1_2 = annotation.get('step1_to_step2_changes', {})

            # Answer changes
            if changes_1_2.get('answer_changed'):
                stats['answer_changed'] += 1

            # Belief changes
            belief1 = step1.get('answer_belief')
            belief2 = step2.get('answer_belief')
            belief_change = calculate_belief_change(belief1, belief2)

            if belief_change is not None:
                stats['belief_changes'].append(belief_change)

            # Persuasion to wrong answer
            step1_correct = step1.get('is_correct', False)
            step2_correct = step2.get('is_correct', False)
            step3_correct = step3.get('is_correct', False)
            answer_changed = changes_1_2.get('answer_changed', False)

            if step1_correct:
                stats['step1_correct'] += 1
            if step2_correct:
                stats['step2_correct'] += 1
            if step3_correct:
                stats['step3_correct'] += 1

            if answer_changed:
                if step1_correct and not step2_correct:
                    stats['correct_to_incorrect'] += 1
                elif not step1_correct and step2_correct:
                    stats['incorrect_to_correct'] += 1

        model_stats[model_key] = stats

    # Print detailed comparison
    print("\n" + "-"*80)
    print("DETAILED MODEL ANALYSIS")
    print("-"*80)

    for model_key, stats in sorted(model_stats.items(), key=lambda x: x[1]['n'], reverse=True):
        print(f"\n{'='*80}")
        print(f"Model: {model_key}")
        print(f"{'='*80}")
        print(f"Total annotations: {stats['n']}")

        # Accuracy analysis
        step1_accuracy = stats['step1_correct'] / stats['n'] * 100 if stats['n'] > 0 else 0
        step2_accuracy = stats['step2_correct'] / stats['n'] * 100 if stats['n'] > 0 else 0
        step3_accuracy = stats['step3_correct'] / stats['n'] * 100 if stats['n'] > 0 else 0

        print(f"\nAccuracy:")
        print(f"  Step 1 (Initial): {stats['step1_correct']}/{stats['n']} ({step1_accuracy:.1f}%)")
        print(f"  Step 2 (After AI): {stats['step2_correct']}/{stats['n']} ({step2_accuracy:.1f}%)")
        print(f"  Step 3 (After truth): {stats['step3_correct']}/{stats['n']} ({step3_accuracy:.1f}%)")
        print(f"  Change Step1→Step2: {step2_accuracy - step1_accuracy:+.1f} percentage points")

        # Answer change rate
        change_rate = stats['answer_changed'] / stats['n'] * 100 if stats['n'] > 0 else 0
        print(f"\nAnswer Changes:")
        print(f"  Changed answer: {stats['answer_changed']}/{stats['n']} ({change_rate:.1f}%)")

        # Persuasion metrics
        if stats['n'] > 0:
            persuade_wrong = stats['correct_to_incorrect'] / stats['n'] * 100
            persuade_right = stats['incorrect_to_correct'] / stats['n'] * 100
            print(f"\nPersuasion Effects:")
            print(f"  Correct → Incorrect: {stats['correct_to_incorrect']}/{stats['n']} ({persuade_wrong:.1f}%)")
            print(f"  Incorrect → Correct: {stats['incorrect_to_correct']}/{stats['n']} ({persuade_right:.1f}%)")

            net_effect = stats['incorrect_to_correct'] - stats['correct_to_incorrect']
            print(f"  Net helpful effect: {net_effect:+d} ({(persuade_right - persuade_wrong):+.1f} percentage points)")

        # Belief changes
        if stats['belief_changes']:
            increases = [x for x in stats['belief_changes'] if x > 0]
            decreases = [x for x in stats['belief_changes'] if x < 0]
            unchanged = [x for x in stats['belief_changes'] if x == 0]

            total = len(stats['belief_changes'])
            print(f"\nBelief Changes:")
            print(f"  Increased: {len(increases)}/{total} ({len(increases)/total*100:.1f}%)")
            if increases:
                print(f"    Mean: {statistics.mean(increases):.3f}, Median: {statistics.median(increases):.3f}")

            print(f"  Decreased: {len(decreases)}/{total} ({len(decreases)/total*100:.1f}%)")
            if decreases:
                print(f"    Mean: {statistics.mean(decreases):.3f}, Median: {statistics.median(decreases):.3f}")

            print(f"  Unchanged: {len(unchanged)}/{total} ({len(unchanged)/total*100:.1f}%)")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY COMPARISON TABLE")
    print("="*80)
    print(f"\n{'Model':<50} {'N':>5} {'Ans Chg':>8} {'C→I':>5} {'I→C':>5} {'Net':>5}")
    print("-"*92)

    for model_key, stats in sorted(model_stats.items(), key=lambda x: x[1]['n'], reverse=True):
        ans_change_pct = stats['answer_changed'] / stats['n'] * 100 if stats['n'] > 0 else 0
        c_to_i_pct = stats['correct_to_incorrect'] / stats['n'] * 100 if stats['n'] > 0 else 0
        i_to_c_pct = stats['incorrect_to_correct'] / stats['n'] * 100 if stats['n'] > 0 else 0
        net = i_to_c_pct - c_to_i_pct

        # Truncate model name if too long
        display_name = model_key[:47] + '...' if len(model_key) > 50 else model_key

        print(f"{display_name:<50} {stats['n']:>5} {ans_change_pct:>7.1f}% {c_to_i_pct:>4.1f}% {i_to_c_pct:>4.1f}% {net:>+5.1f}%")

    print("\nNote:")
    print("  C→I = Correct to Incorrect (harmful persuasion)")
    print("  I→C = Incorrect to Correct (helpful persuasion)")
    print("  Net = I→C - C→I (positive is helpful overall)")


def main():
    """Main comparison function"""
    print("\n" + "="*80)
    print("MODEL COMPARISON ANALYSIS - USMLE SAMPLE")
    print("="*80)
    print("\nComparing persuasion effectiveness across different AI models")

    annotations = load_all_annotations()
    if annotations:
        compare_models(annotations)
    else:
        print("\nNo annotations found in", RESULTS_DIR)

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
