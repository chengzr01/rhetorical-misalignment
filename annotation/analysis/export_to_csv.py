#!/usr/bin/env python3
"""
Export USMLE Sample annotation results to CSV file for further analysis.
"""

import json
import os
import csv


# Results directory
RESULTS_DIR = '../results/usmle_sample'


def export_to_csv(annotations, output_file):
    """Export USMLE annotations to CSV"""
    if not annotations:
        print(f"No annotations to export")
        return

    with open(output_file, 'w', newline='') as f:
        fieldnames = [
            'annotator_id', 'case_id', 'agent_name', 'agent_model',
            'correct_answer', 'correct_answer_idx',
            'step1_answer', 'step1_answer_belief', 'step1_is_correct',
            'step2_answer', 'step2_answer_belief', 'step2_is_correct',
            'step3_answer', 'step3_answer_belief', 'step3_is_correct',
            'answer_changed_1_to_2', 'answer_belief_changed_1_to_2',
            'answer_changed_2_to_3', 'answer_belief_changed_2_to_3',
            'belief_change_1_to_2', 'belief_change_2_to_3',
            'correct_to_incorrect', 'incorrect_to_correct',
            'num_highlights', 'num_highlights_step2',
            'session_start', 'step1_time', 'step2_time', 'step3_time',
            'time_step1_seconds', 'time_step2_seconds', 'time_step3_seconds'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for annotation in annotations:
            step1 = annotation.get('step1', {})
            step2 = annotation.get('step2', {})
            step3 = annotation.get('step3', {})
            changes_1_2 = annotation.get('step1_to_step2_changes', {})
            changes_2_3 = annotation.get('step2_to_step3_changes', {})

            # Calculate belief changes
            def calc_change(before, after):
                if before is None or after is None:
                    return None
                return after - before

            belief_change_1_to_2 = calc_change(step1.get('answer_belief'), step2.get('answer_belief'))
            belief_change_2_to_3 = calc_change(step2.get('answer_belief'), step3.get('answer_belief'))

            # Determine if persuaded to wrong answer
            step1_correct = step1.get('is_correct', False)
            step2_correct = step2.get('is_correct', False)
            answer_changed_1_to_2 = changes_1_2.get('answer_changed', False)

            correct_to_incorrect = answer_changed_1_to_2 and step1_correct and not step2_correct
            incorrect_to_correct = answer_changed_1_to_2 and not step1_correct and step2_correct

            # Count highlights in step2
            highlights = annotation.get('highlights', [])
            num_highlights_step2 = sum(1 for h in highlights if h.get('step') == 'step2')

            # Calculate time spent
            try:
                from datetime import datetime
                start = datetime.fromisoformat(annotation.get('session_start'))
                step1_time = datetime.fromisoformat(annotation.get('step1_time'))
                step2_time = datetime.fromisoformat(annotation.get('step2_time'))
                step3_time = datetime.fromisoformat(annotation.get('step3_time'))

                time_step1 = (step1_time - start).total_seconds()
                time_step2 = (step2_time - step1_time).total_seconds()
                time_step3 = (step3_time - step2_time).total_seconds()
            except:
                time_step1 = time_step2 = time_step3 = None

            row = {
                'annotator_id': annotation.get('annotator_id'),
                'case_id': annotation.get('case_id'),
                'agent_name': annotation.get('agent_name'),
                'agent_model': annotation.get('agent_model'),
                'correct_answer': annotation.get('correct_answer'),
                'correct_answer_idx': annotation.get('correct_answer_idx'),
                'step1_answer': step1.get('answer'),
                'step1_answer_belief': step1.get('answer_belief'),
                'step1_is_correct': step1.get('is_correct'),
                'step2_answer': step2.get('answer'),
                'step2_answer_belief': step2.get('answer_belief'),
                'step2_is_correct': step2.get('is_correct'),
                'step3_answer': step3.get('answer'),
                'step3_answer_belief': step3.get('answer_belief'),
                'step3_is_correct': step3.get('is_correct'),
                'answer_changed_1_to_2': changes_1_2.get('answer_changed'),
                'answer_belief_changed_1_to_2': changes_1_2.get('answer_belief_changed'),
                'answer_changed_2_to_3': changes_2_3.get('answer_changed'),
                'answer_belief_changed_2_to_3': changes_2_3.get('answer_belief_changed'),
                'belief_change_1_to_2': belief_change_1_to_2,
                'belief_change_2_to_3': belief_change_2_to_3,
                'correct_to_incorrect': correct_to_incorrect,
                'incorrect_to_correct': incorrect_to_correct,
                'num_highlights': len(highlights),
                'num_highlights_step2': num_highlights_step2,
                'session_start': annotation.get('session_start'),
                'step1_time': annotation.get('step1_time'),
                'step2_time': annotation.get('step2_time'),
                'step3_time': annotation.get('step3_time'),
                'time_step1_seconds': time_step1,
                'time_step2_seconds': time_step2,
                'time_step3_seconds': time_step3
            }

            writer.writerow(row)

    print(f"Exported {len(annotations)} annotations to {output_file}")


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


def main():
    """Main export function"""
    print("\n" + "="*80)
    print("EXPORTING USMLE SAMPLE ANNOTATIONS TO CSV")
    print("="*80)

    annotations = load_all_annotations()
    if annotations:
        export_to_csv(annotations, 'usmle_sample_annotations.csv')
    else:
        print("\nNo annotations found in", RESULTS_DIR)

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
