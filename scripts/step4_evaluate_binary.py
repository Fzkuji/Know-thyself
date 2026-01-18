"""
Step 4 Binary: Evaluate trained metacognition model (binary classification).

Binary version:
- Only "can" and "cannot" (no "uncertain")
- Temperature=0 for all inference (greedy decoding)
- Single trial per question (num_trials=1)
- 2x2 confusion matrix

Compares model's self-assessment (can/cannot) with actual performance.
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
from src.evaluator import is_correct
from src.label_generator import SYSTEM_PROMPT
from src.multi_gpu_inference import create_inference


def classify_ability_binary(correct: bool) -> str:
    """Binary classification: correct -> can, incorrect -> cannot."""
    return "can" if correct else "cannot"


def parse_judgment_response_binary(response: str) -> str:
    """Parse judgment response to extract binary ability prediction (can/cannot)."""
    response = response.strip().lower()

    # Parse \boxed{} format
    match = re.search(r'\\boxed\{(\w+)\}', response)
    if match:
        answer = match.group(1).lower()
        if answer == "yes":
            return "can"
        else:
            return "cannot"

    # Fallback: check keywords
    if "yes" in response:
        return "can"
    else:
        return "cannot"


def evaluate_binary(
    model_name: str,
    lora_path: str,
    samples: list,
    inference_batch_size: int = 16,
    num_gpus: int = None,
):
    """
    Evaluate metacognition accuracy using binary classification.

    Args:
        model_name: Base model name
        lora_path: Path to LoRA adapter (or "none" for baseline)
        samples: List of samples to evaluate
        inference_batch_size: Batch size for inference
        num_gpus: Number of GPUs to use

    Returns:
        List of evaluation results
    """
    print(f"Model: {model_name}")
    print(f"LoRA: {lora_path}")
    print("Mode: Binary classification (can/cannot), temperature=0, single trial")

    # Create inference instance (temperature=0 for greedy decoding)
    inference = create_inference(
        model_name=model_name,
        inference_batch_size=inference_batch_size,
        temperature=0,  # Greedy decoding for all
        num_gpus=num_gpus,
        lora_path=lora_path if lora_path and lora_path.lower() != "none" else None,
    )

    # Step 1: Predict judgment abilities for all samples
    print("\nStep 1: Predicting judgment abilities (temperature=0)...")
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

    # Parse judgment predictions (binary)
    predicted_abilities = [parse_judgment_response_binary(r) for r in judgment_responses]

    # Clean up judgment inference
    if hasattr(inference, 'shutdown'):
        inference.shutdown()
    del inference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2.0)

    # Step 2: Generate QA responses to determine actual abilities (single trial, temperature=0)
    print("\nStep 2: Generating QA responses (temperature=0, single trial)...")

    # Create new inference for QA (temperature=0 for greedy decoding)
    qa_inference = create_inference(
        model_name=model_name,
        inference_batch_size=inference_batch_size,
        temperature=0,  # Greedy decoding for QA
        num_gpus=num_gpus,
        lora_path=lora_path if lora_path and lora_path.lower() != "none" else None,
    )

    # Build QA prompts (single trial per sample)
    qa_prompts = []
    for sample in samples:
        prompt = f"<|im_start|>system\nAnswer the question concisely and directly.<|im_end|>\n"
        prompt += f"<|im_start|>user\n{sample['question']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        qa_prompts.append(prompt)

    # Generate QA responses
    qa_responses = qa_inference.generate_batch(qa_prompts)

    # Clean up QA inference
    if hasattr(qa_inference, 'shutdown'):
        qa_inference.shutdown()
    del qa_inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 3: Evaluate and compile results
    print("\nStep 3: Evaluating results (binary classification)...")
    results = []
    for i, sample in enumerate(samples):
        response = qa_responses[i]
        gold_answers = sample.get("normalized_answers", sample["answers"])

        # Binary: correct or not
        correct = is_correct(response, gold_answers)
        actual = classify_ability_binary(correct)

        results.append({
            "question": sample["question"],
            "predicted": predicted_abilities[i],
            "actual": actual,
            "correct": correct,
            "response": response,
        })

    return results


def compute_metrics_binary(results):
    """Compute evaluation metrics for binary classification."""
    total = len(results)
    abilities = ["can", "cannot"]

    exact_match = sum(1 for r in results if r["predicted"] == r["actual"])

    # 2x2 confusion matrix
    confusion = {}
    for pred in abilities:
        for actual in abilities:
            confusion[f"{pred}_{actual}"] = sum(
                1 for r in results if r["predicted"] == pred and r["actual"] == actual
            )

    # Count per category
    pred_counts = {a: sum(1 for r in results if r["predicted"] == a) for a in abilities}
    actual_counts = {a: sum(1 for r in results if r["actual"] == a) for a in abilities}

    # QA accuracy (single trial)
    total_correct = sum(1 for r in results if r["correct"])
    qa_accuracy = total_correct / total if total > 0 else 0

    # Precision, Recall, F1 for "can" class
    tp = confusion["can_can"]  # True positive: predicted can, actually can
    fp = confusion["can_cannot"]  # False positive: predicted can, actually cannot
    fn = confusion["cannot_can"]  # False negative: predicted cannot, actually can
    tn = confusion["cannot_cannot"]  # True negative: predicted cannot, actually cannot

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total": total,
        "exact_match": exact_match,
        "exact_match_rate": exact_match / total if total > 0 else 0,
        "confusion": confusion,
        "pred_counts": pred_counts,
        "actual_counts": actual_counts,
        # QA accuracy
        "qa_total_correct": total_correct,
        "qa_accuracy": qa_accuracy,
        # Classification metrics
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--lora_path", type=str, default=str(project_root / "outputs/metacog"))
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--inference_batch_size", type=int, default=16, help="Batch size for inference")

    # GPU params
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs to use for inference (default: all available)")

    # Output
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save metrics JSON (default: auto-generate based on split)")

    args = parser.parse_args()

    print(f"Loading TriviaQA {args.split} split...")
    samples = load_triviaqa(split=args.split, num_samples=args.num_samples)
    print(f"Loaded {len(samples)} samples")

    print(f"\n[Binary Mode] Evaluating model: {args.model}")
    print(f"LoRA path: {args.lora_path}")
    print(f"Inference batch size: {args.inference_batch_size}")
    print("Temperature: 0 (greedy decoding)")
    print("Trials per question: 1")

    results = evaluate_binary(
        model_name=args.model,
        lora_path=args.lora_path,
        samples=samples,
        inference_batch_size=args.inference_batch_size,
        num_gpus=args.num_gpus,
    )

    metrics = compute_metrics_binary(results)

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS (Binary) [{args.split.upper()}]")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Total samples: {metrics['total']}")

    # QA Accuracy
    print(f"\n--- QA Performance ---")
    print(f"QA Accuracy: {metrics['qa_total_correct']}/{metrics['total']} ({metrics['qa_accuracy']*100:.1f}%)")

    # Judgment Accuracy
    print(f"\n--- Judgment Performance ---")
    print(f"Exact match (predicted == actual): {metrics['exact_match']} ({metrics['exact_match_rate']*100:.1f}%)")

    # 2x2 Confusion Matrix
    c = metrics["confusion"]
    print(f"\nConfusion Matrix (2x2):")
    print(f"                    actual_can  actual_cannot")
    print(f"  predicted_can        {c['can_can']:5d}          {c['can_cannot']:5d}")
    print(f"  predicted_cannot     {c['cannot_can']:5d}          {c['cannot_cannot']:5d}")

    # Classification metrics
    print(f"\n--- Classification Metrics (for 'can' class) ---")
    print(f"Precision: {metrics['precision']*100:.1f}%")
    print(f"Recall: {metrics['recall']*100:.1f}%")
    print(f"F1 Score: {metrics['f1']*100:.1f}%")

    # Category counts
    pred = metrics["pred_counts"]
    actual = metrics["actual_counts"]
    print(f"\nPredicted distribution: can={pred['can']}, cannot={pred['cannot']}")
    print(f"Actual distribution:    can={actual['can']}, cannot={actual['cannot']}")

    # Save metrics to JSON
    if args.output_json:
        output_path = Path(args.output_json)
    else:
        # Auto-generate path based on model directory
        model_path = Path(args.model)
        if model_path.exists() and model_path.is_dir():
            output_path = model_path / f"metrics_binary_{args.split}.json"
        else:
            output_path = Path(f"metrics_binary_{args.split}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {output_path}")

    print("Done!")


if __name__ == "__main__":
    main()
