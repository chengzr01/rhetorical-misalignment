#!/usr/bin/env python3
"""
Test language models directly on USMLE sample questions.
This script evaluates models' performance on clinical multiple-choice questions.

Example usage:
    # (same as before)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

# Add parent directory to path to import from agents and interface
sys.path.insert(0, str(Path(__file__).parent.parent))

from interface.client import NvidiaChatClient, OpenRouterChatClient, SGLangChatClient


def load_prompt_template(prompt_path: str | Path) -> str:
    """Load prompt template from YAML file."""
    with open(prompt_path, "r") as f:
        data = yaml.safe_load(f)
    return data["prompt"]


def format_options(options: Dict[str, str]) -> str:
    """Format options dictionary as a string for the prompt."""
    formatted = []
    for key in sorted(options.keys()):
        formatted.append(f"{key}. {options[key]}")
    return "\n".join(formatted)


def build_test_prompt(question: str, options: Dict[str, str], prompt_template: str) -> str:
    """Build the full test prompt with question and options."""
    options_text = format_options(options)

    prompt = prompt_template.replace("<QUESTION>", question)
    prompt = prompt.replace("<OPTIONS>", options_text)

    return prompt


def extract_answer(text: str) -> str | None:
    """
    Extract the answer letter from model's response.

    Looks for pattern: "<answer>letter</answer>" (XML-style format).
    Also supports legacy formats for backward compatibility.
    """
    if not text:
        return None

    # Try to find answer patterns (case insensitive)
    patterns = [
        r'<answer>\s*([A-E])\s*</answer>',  # <answer>A</answer>
        r'<answer>([A-E])</answer>',  # <answer>A</answer> (no whitespace)
        r'ANSWER:\s*([A-E])',  # Legacy: ANSWER: A
        r'ANSWER:\s*([A-E])\.',  # Legacy: ANSWER: A.
        r'answer:\s*([A-E])',  # Legacy: answer: A (lowercase)
        r'\*\*ANSWER:\s*([A-E])',  # Legacy: **ANSWER: A
        r'Final Answer:\s*([A-E])',  # Legacy: Final Answer: A
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def extract_belief(text: str) -> float | None:
    """
    Extract the belief value from model's response.

    Looks for pattern: "<belief>value</belief>" where value is a float between 0 and 1.
    """
    if not text:
        return None

    # Try to find belief pattern (case insensitive)
    patterns = [
        r'<belief>\s*(0?\.\d+|1\.0+|0|1)\s*</belief>',  # <belief>0.95</belief>
        r'<belief>(0?\.\d+|1\.0+|0|1)</belief>',  # <belief>0.95</belief> (no whitespace)
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                belief_value = float(match.group(1))
                # Validate that belief is between 0 and 1
                if 0.0 <= belief_value <= 1.0:
                    return belief_value
            except ValueError:
                continue

    return None


def setup_client(
    backend: str = "nvidia",
    sglang_port: int = 30000,
    sglang_base_url: str = "http://127.0.0.1",
) -> NvidiaChatClient | OpenRouterChatClient | SGLangChatClient:
    """Setup API client based on backend type."""
    if backend == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("Set NVIDIA_API_KEY environment variable")
        return NvidiaChatClient(api_key=api_key)
    elif backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY environment variable")
        return OpenRouterChatClient(
            api_key=api_key, base_url="https://openrouter.ai/api/v1"
        )
    elif backend == "sglang":
        return SGLangChatClient(port=sglang_port, base_url=sglang_base_url)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def test_single_question(
    question_data: Dict[str, Any],
    client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    model: str,
    prompt_template: str,
    temperature: float = 0.0,
    elicit_belief: bool = False,
) -> Dict[str, Any]:
    """
    Test the model on a single question.

    Args:
        question_data: Dictionary containing question, options, answer, etc.
        client: LLM client
        model: Model identifier
        prompt_template: Prompt template string
        temperature: Sampling temperature (default 0.0 for deterministic responses)
        elicit_belief: Whether to extract belief values from responses

    Returns:
        Dictionary with test results
    """
    question = question_data["question"]
    options = question_data["options"]
    correct_answer_idx = question_data["answer_idx"]

    # Build prompt
    prompt = build_test_prompt(question, options, prompt_template)

    # Get model response
    messages = [{"role": "user", "content": prompt}]

    try:
        response = client.create_completion(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    except Exception as e:
        print(f"Error processing question {question_data.get('id', 'unknown')}: {str(e)}")
        result = {
            "id": question_data.get("id"),
            "question": question,
            "options": options,
            "correct_answer": question_data.get("answer"),
            "correct_answer_idx": correct_answer_idx,
            "predicted_answer_idx": None,
            "predicted_answer": None,
            "correct": False,
            "response": None,
            "error": str(e),
            "meta_info": question_data.get("meta_info"),
        }
        if elicit_belief:
            result["belief"] = None
        return result

    # Extract answer
    predicted_answer_idx = extract_answer(response)

    # Extract belief if requested
    belief = None
    if elicit_belief:
        belief = extract_belief(response)

    # Check correctness
    correct = (predicted_answer_idx == correct_answer_idx) if predicted_answer_idx else False

    result = {
        "id": question_data.get("id"),
        "question": question,
        "options": options,
        "correct_answer": question_data.get("answer"),
        "correct_answer_idx": correct_answer_idx,
        "predicted_answer_idx": predicted_answer_idx,
        "predicted_answer": options.get(predicted_answer_idx) if predicted_answer_idx else None,
        "correct": correct,
        "response": response,
        "meta_info": question_data.get("meta_info"),
    }

    if elicit_belief:
        result["belief"] = belief

    return result


def questions_match_cached(
    question_data: Dict[str, Any], cached_result: Dict[str, Any]
) -> bool:
    """Check whether a cached evaluation matches the current question payload."""
    return (
        cached_result.get("question") == question_data.get("question")
        and cached_result.get("options") == question_data.get("options")
        and cached_result.get("correct_answer_idx") == question_data.get("answer_idx")
    )


def run_tests(
    questions: List[Dict[str, Any]],
    client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    model: str,
    prompt_template: str,
    temperature: float = 0.0,
    max_workers: int = 8,
    limit: int | None = None,
    elicit_belief: bool = False,
    existing_results: Dict[str, Dict[str, Any]] | None = None,
    save_interval: int = 0,
    checkpoint_callback: Callable[[List[Dict[str, Any]]], None] | None = None,
) -> List[Dict[str, Any]]:
    """
    Run tests on all questions.

    Args:
        questions: List of question dictionaries
        client: LLM client
        model: Model identifier
        prompt_template: Prompt template string
        temperature: Sampling temperature
        max_workers: Number of parallel workers
        limit: Maximum number of questions to test (None for all)
        elicit_belief: Whether to extract belief values from responses

    Returns:
        List of test results
    """
    if limit:
        questions = questions[:limit]

    existing_results = existing_results or {}
    ordered_ids: List[str] = []
    results_by_id: Dict[str, Dict[str, Any]] = {}
    to_process: List[Dict[str, Any]] = []
    reused = 0

    for question in questions:
        question_id = question.get("id")
        if question_id is None:
            raise ValueError("Each question must have an 'id'")
        question_key = str(question_id)
        ordered_ids.append(question_key)
        cached = existing_results.get(question_key)
        if cached and questions_match_cached(question, cached):
            results_by_id[question_key] = cached
            reused += 1
        else:
            to_process.append(question)

    if to_process:
        print(
            f"Testing {len(to_process)} questions with {max_workers} workers (reused {reused} cached results)..."
        )
    else:
        print(f"Using cached results for all {reused} questions; no API calls needed.")
        return [results_by_id[qid] for qid in ordered_ids]

    def build_error_result(question: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "id": question.get("id"),
            "question": question.get("question"),
            "options": question.get("options"),
            "correct_answer": question.get("answer"),
            "correct_answer_idx": question.get("answer_idx"),
            "predicted_answer_idx": None,
            "predicted_answer": None,
            "correct": False,
            "response": None,
            "error": error_message,
            "meta_info": question.get("meta_info"),
        }
        if elicit_belief:
            result["belief"] = None
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if isinstance(client, SGLangChatClient):

            def job_func(q: Dict[str, Any]) -> Dict[str, Any]:
                thread_client = SGLangChatClient(port=client._port, base_url=client._base_url)
                return test_single_question(
                    q,
                    thread_client,
                    model,
                    prompt_template,
                    temperature,
                    elicit_belief,
                )

        else:

            def job_func(q: Dict[str, Any]) -> Dict[str, Any]:
                return test_single_question(
                    q,
                    client,
                    model,
                    prompt_template,
                    temperature,
                    elicit_belief,
                )

        futures = {
            executor.submit(job_func, question): question for question in to_process
        }

        processed_since_checkpoint = 0
        with tqdm(total=len(futures), desc="Testing") as progress:
            for future in as_completed(futures):
                question = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = build_error_result(question, str(exc))
                result_id = result.get("id")
                if result_id is None:
                    raise ValueError("Each result must include an 'id'")
                result_key = str(result_id)
                results_by_id[result_key] = result
                processed_since_checkpoint += 1
                progress.update(1)
                if (
                    checkpoint_callback is not None
                    and save_interval > 0
                    and processed_since_checkpoint >= save_interval
                ):
                    ordered_results = [
                        results_by_id[qid] for qid in ordered_ids if qid in results_by_id
                    ]
                    checkpoint_callback(ordered_results)
                    processed_since_checkpoint = 0

        if (
            checkpoint_callback is not None
            and save_interval > 0
            and processed_since_checkpoint > 0
        ):
            ordered_results = [results_by_id[qid] for qid in ordered_ids if qid in results_by_id]
            checkpoint_callback(ordered_results)

    missing = [qid for qid in ordered_ids if qid not in results_by_id]
    if missing:
        raise RuntimeError(f"Missing test results for ids: {missing[:10]}")

    return [results_by_id[qid] for qid in ordered_ids]


def calculate_accuracy(results: List[Dict[str, Any]], elicit_belief: bool = False) -> Dict[str, Any]:
    """Calculate accuracy metrics from test results."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    # Calculate by meta_info (step1, step2, step3)
    step1_results = [r for r in results if r.get("meta_info") == "step1"]
    step2_results = [r for r in results if r.get("meta_info") == "step2"]
    step3_results = [r for r in results if r.get("meta_info") == "step3"]

    step1_correct = sum(1 for r in step1_results if r["correct"])
    step2_correct = sum(1 for r in step2_results if r["correct"])
    step3_correct = sum(1 for r in step3_results if r["correct"])

    # Calculate no answer rate
    no_answer = sum(1 for r in results if r["predicted_answer_idx"] is None)

    metrics = {
        "total": total,
        "correct": correct,
        "incorrect": total - correct - no_answer,
        "no_answer": no_answer,
        "accuracy": correct / total if total > 0 else 0.0,
        "step1": {
            "total": len(step1_results),
            "correct": step1_correct,
            "accuracy": step1_correct / len(step1_results) if step1_results else 0.0,
        },
        "step2": {
            "total": len(step2_results),
            "correct": step2_correct,
            "accuracy": step2_correct / len(step2_results) if step2_results else 0.0,
        },
        "step3": {
            "total": len(step3_results),
            "correct": step3_correct,
            "accuracy": step3_correct / len(step3_results) if step3_results else 0.0,
        },
    }

    # Add belief statistics if eliciting beliefs
    if elicit_belief:
        beliefs = [r["belief"] for r in results if r.get("belief") is not None]
        correct_beliefs = [r["belief"] for r in results if r["correct"] and r.get("belief") is not None]
        incorrect_beliefs = [r["belief"] for r in results if not r["correct"] and r.get("belief") is not None]

        metrics["belief"] = {
            "total_with_belief": len(beliefs),
            "no_belief": total - len(beliefs),
            "mean_belief": sum(beliefs) / len(beliefs) if beliefs else 0.0,
            "mean_belief_correct": sum(correct_beliefs) / len(correct_beliefs) if correct_beliefs else 0.0,
            "mean_belief_incorrect": sum(incorrect_beliefs) / len(incorrect_beliefs) if incorrect_beliefs else 0.0,
        }

    return metrics


def print_metrics(metrics: Dict[str, Any], model_name: str, elicit_belief: bool = False) -> None:
    """Print accuracy metrics in a formatted way."""
    print("\n" + "="*80)
    print(f"TEST RESULTS - {model_name}")
    print("="*80)
    print(f"Total questions: {metrics['total']}")
    print(f"Correct: {metrics['correct']} ({metrics['accuracy']*100:.2f}%)")
    print(f"Incorrect: {metrics['incorrect']}")
    print(f"No answer: {metrics['no_answer']}")
    print()
    print(f"Step 1 accuracy: {metrics['step1']['correct']}/{metrics['step1']['total']} "
          f"({metrics['step1']['accuracy']*100:.2f}%)")
    print(f"Step 2 accuracy: {metrics['step2']['correct']}/{metrics['step2']['total']} "
          f"({metrics['step2']['accuracy']*100:.2f}%)")
    print(f"Step 3 accuracy: {metrics['step3']['correct']}/{metrics['step3']['total']} "
          f"({metrics['step3']['accuracy']*100:.2f}%)")

    if elicit_belief and "belief" in metrics:
        print()
        print("BELIEF STATISTICS")
        print("-" * 80)
        belief_stats = metrics["belief"]
        print(f"Questions with belief: {belief_stats['total_with_belief']}/{metrics['total']}")
        print(f"Mean belief (all): {belief_stats['mean_belief']:.3f}")
        print(f"Mean belief (correct): {belief_stats['mean_belief_correct']:.3f}")
        print(f"Mean belief (incorrect): {belief_stats['mean_belief_incorrect']:.3f}")

    print("="*80)


def generate_output_filename(model: str, output_dir: Path, elicit_belief: bool = False) -> Path:
    """
    Generate output filename based on model name.

    Args:
        model: Model identifier (e.g., "meta/llama-3.1-8b-instruct")
        output_dir: Output directory path
        elicit_belief: Whether belief elicitation was used

    Returns:
        Full path to output file
    """
    # Sanitize model name for filename (replace / with -, remove special chars)
    safe_model_name = model.replace("/", "-").replace(":", "-")
    suffix = "_belief" if elicit_belief else ""
    filename = f"test_usmle_sample_{safe_model_name}{suffix}.json"
    return output_dir / filename


def load_cached_results(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load cached evaluation results keyed by question id."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}

    if isinstance(data, dict):
        results = data.get("results") or []
    elif isinstance(data, list):
        results = data
    else:
        return {}

    cache: Dict[str, Dict[str, Any]] = {}
    for item in results:
        qid = item.get("id")
        if qid is None:
            continue
        cache[str(qid)] = item
    return cache


def save_results_json(
    *,
    output_path: Path,
    model: str,
    temperature: float,
    elicit_belief: bool,
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> None:
    payload = {
        "model": model,
        "temperature": temperature,
        "elicit_belief": elicit_belief,
        "metrics": metrics,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def analyze_existing_results(results_path: Path) -> None:
    """
    Load and analyze existing test results from a JSON file.

    Args:
        results_path: Path to the results JSON file
    """
    print(f"Loading existing results from {results_path}...")

    with open(results_path, "r") as f:
        data = json.load(f)

    # Handle both old and new result formats
    if "results" in data:
        # New format with metadata
        results = data.get("results", [])
        model = data.get("model", "Unknown")
        elicit_belief = data.get("elicit_belief", False)
    else:
        # Old format (just a list of results)
        results = data
        model = "Unknown"
        elicit_belief = False

    # Try to extract model name from filename if unknown
    if model == "Unknown":
        filename = results_path.stem  # Get filename without extension
        if filename.startswith("test_usmle_sample_"):
            # Extract model name from filename
            model_part = filename.replace("test_usmle_sample_", "")
            if model_part.endswith("_belief"):
                model_part = model_part[:-7]  # Remove "_belief" suffix
                elicit_belief = True
            model = model_part.replace("-", "/")  # Convert back to model format

    print(f"Loaded {len(results)} results for model: {model}")

    # Recalculate metrics
    metrics = calculate_accuracy(results, elicit_belief=elicit_belief)

    # Print the metrics
    print_metrics(metrics, model, elicit_belief=elicit_belief)

    # Print a few examples
    print("\n" + "="*80)
    print("SAMPLE RESULTS (first 3)")
    print("="*80)
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. Question ID: {result['id']}")
        print(f"   Correct answer: {result['correct_answer_idx']} - {result.get('correct_answer', 'N/A')}")
        print(f"   Predicted: {result['predicted_answer_idx']} - {result.get('predicted_answer', 'N/A')}")
        if elicit_belief and result.get("belief") is not None:
            print(f"   Belief: {result['belief']:.3f}")
        print(f"   Result: {'✓ CORRECT' if result['correct'] else '✗ INCORRECT'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test language models directly on USMLE clinical questions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=str,
        default="experiments/questions/clinical_questions_usmle_sample.json",
        help="Input clinical questions JSON file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="prompts/experiments/test.yaml",
        help="Prompt template YAML file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta/llama-3.1-8b-instruct",
        help="Model to test (e.g., meta/llama-3.1-8b-instruct, meta/llama-3.3-70b-instruct)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nvidia",
        choices=["nvidia", "openrouter", "sglang"],
        help="Server backend"
    )
    parser.add_argument(
        "--sglang-port",
        type=int,
        default=30000,
        help="Port for SGLang server"
    )
    parser.add_argument(
        "--sglang-base-url",
        type=str,
        default="http://127.0.0.1",
        help="Base URL for SGLang server"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for deterministic)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/tests",
        help="Output directory for detailed results"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to test (for debugging)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save partial results every N new evaluations (0 disables)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute all questions even if cached results exist"
    )
    parser.add_argument(
        "--elicit-belief",
        action="store_true",
        help="Elicit and extract belief values from model responses"
    )
    parser.add_argument(
        "--analyze-existing",
        type=str,
        default=None,
        help="Path to existing results JSON file to analyze (skips running new tests)"
    )

    args = parser.parse_args()

    # If analyzing existing results, do that and exit
    if args.analyze_existing:
        analyze_existing_results(Path(args.analyze_existing))
        return

    # Load questions
    print(f"Loading questions from {args.input}...")
    with open(args.input, "r") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    # Load prompt template
    print(f"Loading prompt template from {args.prompt}...")
    prompt_template = load_prompt_template(args.prompt)

    # Setup client
    print(f"Setting up {args.backend} client...")
    client = setup_client(
        backend=args.backend,
        sglang_port=args.sglang_port,
        sglang_base_url=args.sglang_base_url,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = generate_output_filename(
        args.model, output_dir, elicit_belief=args.elicit_belief
    )

    cached_results: Dict[str, Dict[str, Any]] = {}
    if not args.overwrite:
        cached_results = load_cached_results(output_path)
        if cached_results:
            print(f"Loaded {len(cached_results)} cached results from {output_path}")

    def write_checkpoint(partial_results: List[Dict[str, Any]]) -> None:
        metrics_partial = calculate_accuracy(
            partial_results, elicit_belief=args.elicit_belief
        )
        save_results_json(
            output_path=output_path,
            model=args.model,
            temperature=args.temperature,
            elicit_belief=args.elicit_belief,
            results=partial_results,
            metrics=metrics_partial,
        )
        print(
            f"Saved checkpoint with {len(partial_results)} evaluated questions → {output_path}"
        )

    save_interval = max(args.save_interval, 0)
    checkpoint_callback = write_checkpoint if save_interval > 0 else None

    # Run tests
    print(f"\nTesting model: {args.model}")
    print(f"Temperature: {args.temperature}")
    if args.elicit_belief:
        print("Belief elicitation: ENABLED")

    results = run_tests(
        questions=questions,
        client=client,
        model=args.model,
        prompt_template=prompt_template,
        temperature=args.temperature,
        max_workers=args.max_workers,
        limit=args.limit,
        elicit_belief=args.elicit_belief,
        existing_results=None if args.overwrite else cached_results,
        save_interval=save_interval,
        checkpoint_callback=checkpoint_callback,
    )

    # Calculate accuracy
    metrics = calculate_accuracy(results, elicit_belief=args.elicit_belief)

    # Print results
    print_metrics(metrics, args.model, elicit_belief=args.elicit_belief)

    save_results_json(
        output_path=output_path,
        model=args.model,
        temperature=args.temperature,
        elicit_belief=args.elicit_belief,
        results=results,
        metrics=metrics,
    )

    print(f"\nDetailed results saved to: {output_path}")

    # Print a few examples
    print("\n" + "="*80)
    print("SAMPLE RESULTS (first 3)")
    print("="*80)
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. Question ID: {result['id']}")
        print(f"   Correct answer: {result['correct_answer_idx']} - {result.get('correct_answer', 'N/A')}")
        print(f"   Predicted: {result['predicted_answer_idx']} - {result.get('predicted_answer', 'N/A')}")
        if args.elicit_belief and result.get("belief") is not None:
            print(f"   Belief: {result['belief']:.3f}")
        if result.get("error"):
            print(f"   Error: {result['error']}")
        print(f"   Result: {'✓ CORRECT' if result['correct'] else '✗ INCORRECT'}")


if __name__ == "__main__":
    main()
