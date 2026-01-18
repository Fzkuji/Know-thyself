"""
Step 4: Evaluate trained metacognition model on test set.

Compares model's self-assessment (can/uncertain/cannot) with actual performance.
Supports batch inference for better GPU utilization.
Multi-GPU inference is enabled by default when multiple GPUs are available.
"""

import argparse
import sys
import re
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import gc
import time
from tqdm import tqdm

from src.data_loader import load_triviaqa
from src.evaluator import is_correct, classify_ability
from src.label_generator import SYSTEM_PROMPT
from src.multi_gpu_inference import create_inference


def parse_judgment_response(response: str) -> str:
    """Parse judgment response to extract ability prediction."""
    response = response.strip().lower()

    # Parse \boxed{} format
    match = re.search(r'\\boxed\{(\w+)\}', response)
    if match:
        answer = match.group(1).lower()
        if answer == "yes":
            return "can"
        elif answer == "uncertain":
            return "uncertain"
        else:
            return "cannot"

    # Fallback: check keywords
    if "yes" in response:
        return "can"
    elif "uncertain" in response:
        return "uncertain"
    else:
        return "cannot"


def evaluate_with_inference(
    model_name: str,
    lora_path: str,
    samples: list,
    num_trials: int = 5,
    inference_batch_size: int = 16,
    num_gpus: int = None,
):
    """
    Evaluate metacognition accuracy using create_inference.

    Args:
        model_name: Base model name
        lora_path: Path to LoRA adapter (or "none" for baseline)
        samples: List of samples to evaluate
        num_trials: Number of trials per question for actual ability
        inference_batch_size: Batch size for inference
        num_gpus: Number of GPUs to use

    Returns:
        List of evaluation results
    """
    print(f"Model: {model_name}")
    print(f"LoRA: {lora_path}")

    # Create inference instance (with or without LoRA, auto multi-GPU)
    inference = create_inference(
        model_name=model_name,
        inference_batch_size=inference_batch_size,
        temperature=0,  # Greedy decoding for judgment prediction
        num_gpus=num_gpus,
        lora_path=lora_path if lora_path and lora_path.lower() != "none" else None,
    )

    # Step 1: Predict judgment abilities for all samples
    print("\nStep 1: Predicting judgment abilities...")
    print(f"Building {len(samples)} judgment prompts...")
    judgment_prompts = []
    for sample in samples:
        # Build judgment prompt
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        prompt += f"<|im_start|>user\nCan you answer this question correctly?\n\nQuestion: {sample['question']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        judgment_prompts.append(prompt)

    # Generate judgment predictions (batch inference for all samples)
    print("Running batch inference for judgment...")
    judgment_responses = inference.generate_batch(judgment_prompts)

    # Parse judgment predictions
    predicted_abilities = [parse_judgment_response(r) for r in judgment_responses]

    # Clean up judgment inference - ensure complete release before creating new instance
    if hasattr(inference, 'shutdown'):
        inference.shutdown()
    del inference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2.0)  # Wait for GPU memory to be fully released

    # Step 2: Generate QA responses to determine actual abilities
    print("\nStep 2: Generating QA responses to determine actual abilities...")

    # Create new inference for QA (higher temperature for diversity, auto multi-GPU)
    qa_inference = create_inference(
        model_name=model_name,
        inference_batch_size=inference_batch_size,
        temperature=1.0,  # Higher temperature for QA diversity
        num_gpus=num_gpus,
        lora_path=lora_path if lora_path and lora_path.lower() != "none" else None,
    )

    # Build QA prompts (num_trials per sample)
    qa_prompts = []
    for sample in samples:
        prompt = f"<|im_start|>system\nAnswer the question concisely and directly.<|im_end|>\n"
        prompt += f"<|im_start|>user\n{sample['question']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        qa_prompts.extend([prompt] * num_trials)

    # Generate QA responses
    all_qa_responses = qa_inference.generate_batch(qa_prompts)

    # Clean up QA inference
    if hasattr(qa_inference, 'shutdown'):
        qa_inference.shutdown()
    del qa_inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 3: Evaluate and compile results
    print("\nStep 3: Evaluating results...")
    results = []
    for i, sample in enumerate(samples):
        # Get QA responses for this sample
        start_idx = i * num_trials
        end_idx = start_idx + num_trials
        responses = all_qa_responses[start_idx:end_idx]

        # Evaluate correctness
        gold_answers = sample.get("normalized_answers", sample["answers"])
        correct_count = sum(1 for r in responses if is_correct(r, gold_answers))

        accuracy = correct_count / num_trials
        actual = classify_ability(correct_count, num_trials)

        results.append({
            "question": sample["question"],
            "predicted": predicted_abilities[i],
            "actual": actual,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "responses": responses,
        })

    return results


def compute_metrics(results):
    """Compute evaluation metrics."""
    total = len(results)
    abilities = ["can", "uncertain", "cannot"]

    exact_match = sum(1 for r in results if r["predicted"] == r["actual"])

    # 3x3 confusion matrix
    confusion = {}
    for pred in abilities:
        for actual in abilities:
            confusion[f"{pred}_{actual}"] = sum(
                1 for r in results if r["predicted"] == pred and r["actual"] == actual
            )

    # Count per category
    pred_counts = {a: sum(1 for r in results if r["predicted"] == a) for a in abilities}
    actual_counts = {a: sum(1 for r in results if r["actual"] == a) for a in abilities}

    # QA accuracy statistics
    total_correct = sum(r["correct_count"] for r in results)
    total_trials = sum(len(r["responses"]) for r in results)
    avg_accuracy = sum(r["accuracy"] for r in results) / total if total > 0 else 0

    return {
        "total": total,
        "exact_match": exact_match,
        "exact_match_rate": exact_match / total if total > 0 else 0,
        "confusion": confusion,
        "pred_counts": pred_counts,
        "actual_counts": actual_counts,
        # QA accuracy
        "qa_total_correct": total_correct,
        "qa_total_trials": total_trials,
        "qa_accuracy": total_correct / total_trials if total_trials > 0 else 0,
        "qa_avg_accuracy": avg_accuracy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--lora_path", type=str, default=str(project_root / "outputs/metacog"))
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--num_trials", type=int, default=5, help="Trials per question for actual ability")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--inference_batch_size", type=int, default=16, help="Batch size for inference")

    # GPU params
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs to use for inference (default: all available)")

    # Output
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save metrics JSON (default: auto-generate based on split)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Epoch number (for display purposes)")

    args = parser.parse_args()

    print(f"Loading TriviaQA {args.split} split...")
    samples = load_triviaqa(split=args.split, num_samples=args.num_samples)
    print(f"Loaded {len(samples)} samples")

    print(f"\nEvaluating model: {args.model}")
    print(f"LoRA path: {args.lora_path}")
    print(f"Inference batch size: {args.inference_batch_size}")

    results = evaluate_with_inference(
        model_name=args.model,
        lora_path=args.lora_path,
        samples=samples,
        num_trials=args.num_trials,
        inference_batch_size=args.inference_batch_size,
        num_gpus=args.num_gpus,
    )

    metrics = compute_metrics(results)

    print("\n" + "=" * 60)
    epoch_str = f" - Epoch {args.epoch}" if args.epoch is not None else ""
    print(f"EVALUATION RESULTS [{args.split.upper()}]{epoch_str}")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Total samples: {metrics['total']}")

    # QA Accuracy (how well the model actually answers questions)
    print(f"\n--- QA Performance ---")
    print(f"QA Accuracy: {metrics['qa_total_correct']}/{metrics['qa_total_trials']} ({metrics['qa_accuracy']*100:.1f}%)")
    print(f"Average per-question accuracy: {metrics['qa_avg_accuracy']*100:.1f}%")

    # Judgment Accuracy (how well the model predicts its own ability)
    print(f"\n--- Judgment Performance ---")
    print(f"Exact match (predicted == actual): {metrics['exact_match']} ({metrics['exact_match_rate']*100:.1f}%)")

    # 3x3 Confusion Matrix
    c = metrics["confusion"]
    print(f"\nConfusion Matrix:")
    print(f"                      actual_can  actual_uncertain  actual_cannot")
    print(f"  predicted_can          {c['can_can']:5d}           {c['can_uncertain']:5d}            {c['can_cannot']:5d}")
    print(f"  predicted_uncertain    {c['uncertain_can']:5d}           {c['uncertain_uncertain']:5d}            {c['uncertain_cannot']:5d}")
    print(f"  predicted_cannot       {c['cannot_can']:5d}           {c['cannot_uncertain']:5d}            {c['cannot_cannot']:5d}")

    # Category counts
    pred = metrics["pred_counts"]
    actual = metrics["actual_counts"]
    print(f"\nPredicted distribution: can={pred['can']}, uncertain={pred['uncertain']}, cannot={pred['cannot']}")
    print(f"Actual distribution:    can={actual['can']}, uncertain={actual['uncertain']}, cannot={actual['cannot']}")

    # Save metrics to JSON
    if args.output_json:
        output_path = Path(args.output_json)
    else:
        # Auto-generate path based on model directory
        model_path = Path(args.model)
        if model_path.exists() and model_path.is_dir():
            output_path = model_path / f"metrics_{args.split}.json"
        else:
            output_path = Path(f"metrics_{args.split}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {output_path}")

    print("Done!")


if __name__ == "__main__":
    main()
