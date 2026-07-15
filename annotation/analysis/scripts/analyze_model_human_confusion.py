#!/usr/bin/env python3
"""Build a confusion matrix relating human and model correctness transitions."""

import argparse
import json
import os
from collections import Counter


DEFAULT_MODEL_CONFIG = {
    'gpt': {
        'agent_model': 'openai/gpt-5.1',
        'test_file': 'test_usmle_sample_openai-gpt-5.1_belief.json',
    },
    'gemini': {
        'agent_model': 'google/gemini-2.5-pro',
        'test_file': 'test_usmle_sample_google-gemini-2.5-pro_belief.json',
    },
    'claude': {
        'agent_model': 'anthropic/claude-haiku-4.5',
        'test_file': 'test_usmle_sample_anthropic-claude-haiku-4.5_belief.json',
    },
    'deepseek': {
        'agent_model': 'deepseek/deepseek-chat-v3.1',
        'test_file': 'test_usmle_sample_deepseek-deepseek-chat-v3.1_belief.json',
    },
    'llama': {
        'agent_model': 'meta-llama/llama-3.3-70b-instruct',
        'test_file': 'test_usmle_sample_meta-llama-llama-3.3-70b-instruct_belief.json',
    },
    'llama-small': {
        'agent_model': 'meta-llama/llama-3.1-8b-instruct',
        'test_file': 'test_usmle_sample_meta-llama-llama-3.1-8b-instruct_belief.json',
    },
    'llama-dpo': {
        'agent_model': 'allenai/Llama-3.1-Tulu-3-8B-DPO',
        'test_file': 'test_usmle_sample_allenai-Llama-3.1-Tulu-3-8B-DPO_belief.json',
    },
    'llama-sft': {
        'agent_model': 'allenai/Llama-3.1-Tulu-3-8B-SFT',
        'test_file': 'test_usmle_sample_allenai-Llama-3.1-Tulu-3-8B-SFT_belief.json',
    },
}


def normalise_keys(raw_keys: list[str] | None) -> list[str]:
    """Parse and validate model key selections."""

    if not raw_keys:
        return list(DEFAULT_MODEL_CONFIG.keys())

    keys: list[str] = []
    for raw in raw_keys:
        for part in raw.split(','):
            part = part.strip()
            if not part:
                continue
            if part not in DEFAULT_MODEL_CONFIG:
                known = ', '.join(sorted(DEFAULT_MODEL_CONFIG))
                raise ValueError(f"Unknown model key '{part}'. Known keys: {known}")
            keys.append(part)

    if not keys:
        raise ValueError("No valid model keys supplied.")

    seen = set()
    deduped: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)

    return deduped


ROW_LABELS = {False: 'Initially incorrect', True: 'Initially correct'}
COL_HEADERS = ['Revised incorrect (%)', 'Revised correct (%)', 'Row share (%)']
COL_WIDTHS = [26, 26, 18, 15]


def load_annotations(results_dir: str) -> list:
    annotations = []
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        if not name.endswith('.json') or not os.path.isfile(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                annotations.append(json.load(handle))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: skipping {name}: {exc}")
    return annotations


def load_model_results(tests_dir: str, selected_config: dict[str, dict]) -> dict:
    model_case_map = {}
    if not os.path.exists(tests_dir):
        raise FileNotFoundError(f"Tests directory not found: {tests_dir}")

    for key, cfg in selected_config.items():
        test_file = cfg.get('test_file')
        if not test_file:
            print(f"Warning: model key '{key}' missing test_file configuration; skipping")
            continue

        path = os.path.join(tests_dir, test_file)
        if not os.path.isfile(path):
            print(f"Warning: test results not found for '{key}' at {test_file}")
            continue

        try:
            with open(path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: could not load {test_file}: {exc}")
            continue

        expected_model = cfg['agent_model']
        data_model = data.get('model')
        if data_model and data_model != expected_model:
            print(
                f"Warning: model mismatch for '{key}': results report '{data_model}' but"
                f" expected '{expected_model}'. Using expected name."
            )

        cases = {}
        for entry in data.get('results', []):
            case_id = entry.get('id')
            correct = entry.get('correct')
            if case_id and isinstance(correct, bool):
                cases[case_id] = correct

        if not cases:
            print(f"Warning: no result entries found in {test_file}")
            continue

        model_case_map[expected_model] = cases

    return model_case_map


def new_matrix() -> dict:
    return {
        True: {
            False: {False: 0, True: 0},
            True: {False: 0, True: 0},
        },
        False: {
            False: {False: 0, True: 0},
            True: {False: 0, True: 0},
        },
    }


def compute_matrix(
    annotations: list,
    model_case_map: dict,
    allowed_models: set[str],
) -> tuple[dict, dict, Counter, int, Counter]:
    overall_matrix = new_matrix()
    per_model_matrices: dict = {}
    skipped = Counter()
    included = 0
    per_model_counts: Counter = Counter()

    for ann in annotations:
        model_name = ann.get('agent_model')
        case_id = ann.get('case_id')
        if not model_name:
            skipped['missing_model_name'] += 1
            continue
        if model_name not in allowed_models:
            skipped['model_filtered_out'] += 1
            continue
        if model_name not in model_case_map:
            skipped['model_results_missing'] += 1
            continue
        if not case_id:
            skipped['missing_case_id'] += 1
            continue

        step1 = ann.get('step1', {})
        step2 = ann.get('step2', {})
        initial_correct = step1.get('is_correct')
        revised_correct = step2.get('is_correct')
        if not isinstance(initial_correct, bool) or not isinstance(revised_correct, bool):
            skipped['missing_step_correctness'] += 1
            continue

        model_correct = model_case_map[model_name].get(case_id)
        if not isinstance(model_correct, bool):
            skipped['model_case_missing'] += 1
            continue

        overall_matrix[model_correct][initial_correct][revised_correct] += 1

        model_matrix = per_model_matrices.setdefault(model_name, new_matrix())
        model_matrix[model_correct][initial_correct][revised_correct] += 1

        per_model_counts[model_name] += 1
        included += 1

    return overall_matrix, per_model_matrices, skipped, included, per_model_counts


def render_report(
    overall_matrix: dict,
    per_model_matrices: dict,
    per_model_counts: Counter,
    included: int,
    skipped: Counter,
    model_keys: list[str],
    selected_config: dict[str, dict],
) -> str:
    def header_row(title: str) -> list[str]:
        return [
            f"| {title.ljust(COL_WIDTHS[0])} | {COL_HEADERS[0].ljust(COL_WIDTHS[1])} | "
            f"{COL_HEADERS[1].ljust(COL_WIDTHS[2])} | {COL_HEADERS[2].ljust(COL_WIDTHS[3])} |",
            f"| {'-' * COL_WIDTHS[0]} | {'-' * COL_WIDTHS[1]} | {'-' * COL_WIDTHS[2]} | {'-' * COL_WIDTHS[3]} |",
        ]

    def format_cell(count: int, subtotal: int) -> str:
        pct = (count / subtotal * 100.0) if subtotal else 0.0
        return f"{pct:5.1f}%"

    def format_matrix(matrix: dict) -> list[str]:
        block: list[str] = []
        subtotals = {
            True: sum(
                matrix[True][initial][revised]
                for initial in (False, True)
                for revised in (False, True)
            ),
            False: sum(
                matrix[False][initial][revised]
                for initial in (False, True)
                for revised in (False, True)
            ),
        }
        for model_correct in (True, False):
            title = 'Model correct' if model_correct else 'Model incorrect'
            block.extend(header_row(title))
            subtotal = subtotals[model_correct]
            for initial_correct in (False, True):
                row_total = (
                    matrix[model_correct][initial_correct][False]
                    + matrix[model_correct][initial_correct][True]
                )
                cell_left = format_cell(
                    matrix[model_correct][initial_correct][False],
                    subtotal,
                )
                cell_right = format_cell(
                    matrix[model_correct][initial_correct][True],
                    subtotal,
                )
                row_pct = (row_total / subtotal * 100.0) if subtotal else 0.0
                total_str = f"{row_pct:5.1f}%"
                block.append(
                    f"| {ROW_LABELS[initial_correct].ljust(COL_WIDTHS[0])} | "
                    f"{cell_left.ljust(COL_WIDTHS[1])} | {cell_right.ljust(COL_WIDTHS[2])} | "
                    f"{total_str.ljust(COL_WIDTHS[3])} |"
                )
            block.append("")
        return block

    lines: list[str] = []
    lines.append("MAIN CORRECTNESS TRANSITION MATRIX")
    lines.append(
        "Included model keys: "
        + ", ".join(model_keys)
    )
    lines.append(
        "Agent models: "
        + ", ".join(selected_config[key]['agent_model'] for key in model_keys)
    )
    lines.append("")
    model_correct_total = sum(
        overall_matrix[True][initial][revised]
        for initial in (False, True)
        for revised in (False, True)
    )
    model_incorrect_total = sum(
        overall_matrix[False][initial][revised]
        for initial in (False, True)
        for revised in (False, True)
    )
    overall_total = model_correct_total + model_incorrect_total
    model_correct_pct = (model_correct_total / overall_total * 100.0) if overall_total else 0.0
    model_incorrect_pct = (model_incorrect_total / overall_total * 100.0) if overall_total else 0.0
    lines.append(f"Model correct share: {model_correct_pct:.1f}%")
    lines.append(f"Model incorrect share: {model_incorrect_pct:.1f}%")
    lines.append("")

    lines.extend(format_matrix(overall_matrix))

    lines.append(f"Included annotations: {included}")
    skipped_total = sum(skipped.values())
    lines.append(f"Skipped annotations: {skipped_total}")
    for reason, count in skipped.most_common():
        lines.append(f"  - {reason}: {count}")

    lines.append("")
    lines.append("PER-MODEL CONFUSION MATRICES")
    lines.append("")

    for key in model_keys:
        cfg = selected_config[key]
        agent_model = cfg['agent_model']
        lines.append(f"Model key: {key}")
        lines.append(f"Agent model: {agent_model}")
        count = per_model_counts.get(agent_model, 0)
        lines.append(f"Annotations included: {count}")
        model_matrix = per_model_matrices.get(agent_model)
        if not model_matrix:
            lines.append("(No annotations available for this model.)")
            lines.append("")
            continue
        block = format_matrix(model_matrix)
        lines.extend(block)
        if block and block[-1] != "":
            lines.append("")

    return "\n".join(lines)


def resolve_path(value: str | None, fallback: str) -> str:
    if value:
        return os.path.abspath(value)
    return os.path.abspath(fallback)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze human/model correctness transitions.")
    parser.add_argument('--results-dir', default=None, help='Annotation results directory')
    parser.add_argument('--tests-dir', default=None, help='Model test results directory')
    parser.add_argument('--output', default=None, help='Optional report file path')
    parser.add_argument('--no-write', action='store_true', help='Skip writing the report to disk')
    parser.add_argument(
        '--model-keys',
        action='append',
        help=(
            'Restrict analysis to these model keys (comma-separated). '
            f"Available defaults: {', '.join(sorted(DEFAULT_MODEL_CONFIG))}"
        ),
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.normpath(os.path.join(script_dir, '../../results/usmle_sample'))
    default_tests = os.path.normpath(os.path.join(script_dir, '../../../experiments/tests'))
    default_output = os.path.normpath(os.path.join(script_dir, '../outputs/model_human_confusion_matrix.txt'))

    results_dir = resolve_path(args.results_dir, default_results)
    tests_dir = resolve_path(args.tests_dir, default_tests)
    output_path = resolve_path(args.output, default_output) if not args.no_write else None

    try:
        model_keys = normalise_keys(args.model_keys)
    except ValueError as exc:
        parser.error(str(exc))

    selected_config = {key: DEFAULT_MODEL_CONFIG[key] for key in model_keys}
    allowed_models = {cfg['agent_model'] for cfg in selected_config.values()}

    annotations = load_annotations(results_dir)
    model_case_map = load_model_results(tests_dir, selected_config)
    overall_matrix, per_model_matrices, skipped, included, per_model_counts = compute_matrix(
        annotations,
        model_case_map,
        allowed_models,
    )
    report = render_report(
        overall_matrix,
        per_model_matrices,
        per_model_counts,
        included,
        skipped,
        model_keys,
        selected_config,
    )

    print(report)

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as handle:
            handle.write(report)


if __name__ == '__main__':
    main()
