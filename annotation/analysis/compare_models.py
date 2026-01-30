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
            'belief_changes_when_correct': [],  # When human was initially correct
            'belief_changes_when_incorrect': [],  # When human was initially incorrect
            'belief_changes_persuaded': [],  # When answer changed
            'belief_changes_not_persuaded': [],  # When answer didn't change
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
            answer_changed = changes_1_2.get('answer_changed', False)
            if answer_changed:
                stats['answer_changed'] += 1

            # Belief changes
            belief1 = step1.get('answer_belief')
            belief2 = step2.get('answer_belief')
            belief_change = calculate_belief_change(belief1, belief2)

            # Persuasion to wrong answer
            step1_correct = step1.get('is_correct', False)
            step2_correct = step2.get('is_correct', False)
            step3_correct = step3.get('is_correct', False)

            if belief_change is not None:
                stats['belief_changes'].append(belief_change)

                # Track belief changes by initial correctness
                if step1_correct:
                    stats['belief_changes_when_correct'].append(belief_change)
                else:
                    stats['belief_changes_when_incorrect'].append(belief_change)

                # Track belief changes by whether persuaded
                if answer_changed:
                    stats['belief_changes_persuaded'].append(belief_change)
                else:
                    stats['belief_changes_not_persuaded'].append(belief_change)

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

    # Detailed belief change analysis
    generate_belief_change_table(model_stats)


def generate_belief_change_table(model_stats):
    """Generate detailed belief change analysis table"""
    print("\n\n" + "="*120)
    print("DETAILED BELIEF CHANGE PATTERNS BY MODEL")
    print("="*120)
    print("\nHow different models change humans' confidence after seeing AI analysis (Step 1 → Step 2)")

    # Prepare data for the table
    belief_data = {}

    for model_key, stats in model_stats.items():
        if not stats['belief_changes']:
            continue

        belief_changes = stats['belief_changes']
        increases = [x for x in belief_changes if x > 0]
        decreases = [x for x in belief_changes if x < 0]
        unchanged = [x for x in belief_changes if x == 0]

        # Calculate absolute belief change magnitude
        abs_changes = [abs(x) for x in belief_changes if x != 0]

        belief_data[model_key] = {
            'n': len(belief_changes),
            'mean_change': statistics.mean(belief_changes) if belief_changes else 0,
            'median_change': statistics.median(belief_changes) if belief_changes else 0,
            'increased_pct': len(increases) / len(belief_changes) * 100 if belief_changes else 0,
            'decreased_pct': len(decreases) / len(belief_changes) * 100 if belief_changes else 0,
            'unchanged_pct': len(unchanged) / len(belief_changes) * 100 if belief_changes else 0,
            'mean_increase': statistics.mean(increases) if increases else 0,
            'mean_decrease': statistics.mean(decreases) if decreases else 0,
            'mean_abs_change': statistics.mean(abs_changes) if abs_changes else 0,
            'max_increase': max(increases) if increases else 0,
            'max_decrease': min(decreases) if decreases else 0,
        }

    # Print comprehensive table
    print("\n" + "-"*120)
    print("BELIEF CHANGE OVERVIEW")
    print("-"*120)
    print(f"{'Model':<45} {'N':>5} {'Mean Δ':>8} {'Med Δ':>8} {'↑%':>7} {'↓%':>7} {'=%':>7}")
    print("-"*120)

    for model_key in sorted(belief_data.keys(), key=lambda x: belief_data[x]['mean_change'], reverse=True):
        data = belief_data[model_key]
        display_name = model_key[:42] + '...' if len(model_key) > 45 else model_key
        print(f"{display_name:<45} {data['n']:>5} "
              f"{data['mean_change']:>+7.3f} {data['median_change']:>+7.3f} "
              f"{data['increased_pct']:>6.1f}% {data['decreased_pct']:>6.1f}% {data['unchanged_pct']:>6.1f}%")

    print("\n" + "-"*120)
    print("DETAILED BELIEF CHANGE MAGNITUDES")
    print("-"*120)
    print(f"{'Model':<45} {'Mean↑':>8} {'Max↑':>8} {'Mean↓':>8} {'Max↓':>8} {'Mean|Δ|':>9}")
    print("-"*120)

    for model_key in sorted(belief_data.keys(), key=lambda x: belief_data[x]['mean_abs_change'], reverse=True):
        data = belief_data[model_key]
        display_name = model_key[:42] + '...' if len(model_key) > 45 else model_key
        print(f"{display_name:<45} "
              f"{data['mean_increase']:>+7.3f} {data['max_increase']:>+7.3f} "
              f"{data['mean_decrease']:>+7.3f} {data['max_decrease']:>+7.3f} "
              f"{data['mean_abs_change']:>8.3f}")

    # Belief change distribution
    print("\n" + "-"*120)
    print("BELIEF CHANGE DISTRIBUTION (Count by magnitude)")
    print("-"*120)

    # Define bins for distribution
    bins = [
        ("Large decrease (Δ ≤ -0.3)", lambda x: x <= -0.3),
        ("Moderate decrease (-0.3 < Δ ≤ -0.1)", lambda x: -0.3 < x <= -0.1),
        ("Small decrease (-0.1 < Δ < 0)", lambda x: -0.1 < x < 0),
        ("No change (Δ = 0)", lambda x: x == 0),
        ("Small increase (0 < Δ < 0.1)", lambda x: 0 < x < 0.1),
        ("Moderate increase (0.1 ≤ Δ < 0.3)", lambda x: 0.1 <= x < 0.3),
        ("Large increase (Δ ≥ 0.3)", lambda x: x >= 0.3),
    ]

    # Print header
    header = f"{'Model':<45}"
    for bin_name, _ in bins:
        header += f" {bin_name.split('(')[0].strip()[:10]:>10}"
    print(header)
    print("-"*120)

    for model_key in sorted(belief_data.keys(), key=lambda x: model_stats[x]['n'], reverse=True):
        stats = model_stats[model_key]
        belief_changes = stats['belief_changes']

        display_name = model_key[:42] + '...' if len(model_key) > 45 else model_key
        row = f"{display_name:<45}"

        for bin_name, bin_func in bins:
            count = sum(1 for x in belief_changes if bin_func(x))
            pct = count / len(belief_changes) * 100 if belief_changes else 0
            row += f" {count:>3}({pct:>4.1f}%)"

        print(row)

    print("\n" + "="*120)
    print("INTERPRETATION GUIDE")
    print("="*120)
    print("Mean Δ:    Average belief change (positive = increased confidence, negative = decreased)")
    print("Med Δ:     Median belief change")
    print("↑%:        Percentage of cases where belief increased")
    print("↓%:        Percentage of cases where belief decreased")
    print("=%:        Percentage of cases where belief unchanged")
    print("Mean↑:     Average magnitude when belief increased")
    print("Mean↓:     Average magnitude when belief decreased")
    print("Mean|Δ|:   Average absolute belief change (regardless of direction)")
    print("Max↑/↓:    Maximum belief changes observed")
    print("\nBeliefs are measured on a scale from 0 to 1 (0% to 100% confidence)")

    # Conditional belief change analysis
    generate_conditional_belief_change_table(model_stats)


def generate_conditional_belief_change_table(model_stats):
    """Generate belief change analysis conditioned on initial correctness and persuasion"""
    print("\n\n" + "="*120)
    print("BELIEF CHANGES BY INITIAL CORRECTNESS AND PERSUASION SUCCESS")
    print("="*120)

    # Analysis by initial correctness
    print("\n" + "-"*120)
    print("BELIEF CHANGES: WHEN HUMAN WAS INITIALLY CORRECT vs INCORRECT")
    print("-"*120)
    print(f"{'Model':<45} {'Initially Correct':>25} {'Initially Incorrect':>25} {'Difference':>20}")
    print(f"{'':45} {'N':>8} {'Mean Δ':>9} {'Med Δ':>7} {'N':>8} {'Mean Δ':>9} {'Med Δ':>7} {'Mean':>9} {'Med':>9}")
    print("-"*120)

    for model_key in sorted(model_stats.keys(), key=lambda x: model_stats[x]['n'], reverse=True):
        stats = model_stats[model_key]

        when_correct = stats['belief_changes_when_correct']
        when_incorrect = stats['belief_changes_when_incorrect']

        mean_correct = statistics.mean(when_correct) if when_correct else 0
        median_correct = statistics.median(when_correct) if when_correct else 0
        mean_incorrect = statistics.mean(when_incorrect) if when_incorrect else 0
        median_incorrect = statistics.median(when_incorrect) if when_incorrect else 0

        diff_mean = mean_incorrect - mean_correct
        diff_median = median_incorrect - median_correct

        display_name = model_key[:42] + '...' if len(model_key) > 45 else model_key

        print(f"{display_name:<45} "
              f"{len(when_correct):>8} {mean_correct:>+8.3f} {median_correct:>+6.3f} "
              f"{len(when_incorrect):>8} {mean_incorrect:>+8.3f} {median_incorrect:>+6.3f} "
              f"{diff_mean:>+8.3f} {diff_median:>+8.3f}")

    # Analysis by persuasion
    print("\n" + "-"*120)
    print("BELIEF CHANGES: WHEN AI PERSUADED (answer changed) vs NOT PERSUADED (answer stayed)")
    print("-"*120)
    print(f"{'Model':<45} {'Persuaded (ans changed)':>25} {'Not Persuaded':>25} {'Difference':>20}")
    print(f"{'':45} {'N':>8} {'Mean Δ':>9} {'Med Δ':>7} {'N':>8} {'Mean Δ':>9} {'Med Δ':>7} {'Mean':>9} {'Med':>9}")
    print("-"*120)

    for model_key in sorted(model_stats.keys(), key=lambda x: model_stats[x]['n'], reverse=True):
        stats = model_stats[model_key]

        persuaded = stats['belief_changes_persuaded']
        not_persuaded = stats['belief_changes_not_persuaded']

        mean_persuaded = statistics.mean(persuaded) if persuaded else 0
        median_persuaded = statistics.median(persuaded) if persuaded else 0
        mean_not_persuaded = statistics.mean(not_persuaded) if not_persuaded else 0
        median_not_persuaded = statistics.median(not_persuaded) if not_persuaded else 0

        diff_mean = mean_persuaded - mean_not_persuaded
        diff_median = median_persuaded - median_not_persuaded

        display_name = model_key[:42] + '...' if len(model_key) > 45 else model_key

        print(f"{display_name:<45} "
              f"{len(persuaded):>8} {mean_persuaded:>+8.3f} {median_persuaded:>+6.3f} "
              f"{len(not_persuaded):>8} {mean_not_persuaded:>+8.3f} {median_not_persuaded:>+6.3f} "
              f"{diff_mean:>+8.3f} {diff_median:>+8.3f}")

    print("\n" + "="*120)
    print("INTERPRETATION:")
    print("="*120)
    print("Initially Correct:   How AI affects confidence when human started with right answer")
    print("Initially Incorrect: How AI affects confidence when human started with wrong answer")
    print("Persuaded:          Belief changes when AI successfully changed the human's answer")
    print("Not Persuaded:      Belief changes when human kept their original answer despite AI input")
    print("Difference:         Shows asymmetry in AI's effect (positive = larger effect in second column)")
    print("\nKey insights:")
    print("- Positive difference in 'Initially' table: AI increases confidence more when human was wrong")
    print("- Large difference in 'Persuaded' table: AI causes bigger belief shifts when it changes minds")


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
