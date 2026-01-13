"""
Step 4: Evaluate trained metacognition model on test set.

Compares model's self-assessment (can/uncertain/cannot) with actual performance.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from src.data_loader import load_triviaqa
from src.evaluator import is_correct, classify_ability


def load_trained_model(base_model: str, lora_path: str):
    """Load base model with LoRA weights."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()

    return model, tokenizer


def predict_ability(model, tokenizer, question: str) -> str:
    """Ask trained model to assess its ability to answer."""
    prompt = f"Before answering, assess your ability to answer this question:\n{question}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.1,  # Low temperature for consistent prediction
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip().lower()

    # Parse response to ability
    if "i can answer" in response or "can answer" in response:
        return "can"
    elif "uncertain" in response:
        return "uncertain"
    else:
        return "cannot"


def answer_question(model, tokenizer, question: str) -> str:
    """Get model's actual answer to the question."""
    prompt = f"Question: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    return response


def evaluate_metacognition(
    model,
    tokenizer,
    samples,
    num_trials: int = 5,
):
    """
    Evaluate metacognition accuracy.

    For each question:
    1. Get model's self-assessment (predicted ability)
    2. Test actual ability (answer N times, compute accuracy)
    3. Compare prediction vs reality
    """
    results = []

    for sample in tqdm(samples, desc="Evaluating"):
        question = sample["question"]
        gold_answers = sample.get("normalized_answers", sample["answers"])

        # 1. Get predicted ability
        predicted = predict_ability(model, tokenizer, question)

        # 2. Test actual ability
        correct_count = 0
        responses = []
        for _ in range(num_trials):
            response = answer_question(model, tokenizer, question)
            responses.append(response)
            if is_correct(response, gold_answers):
                correct_count += 1

        accuracy = correct_count / num_trials
        actual = classify_ability(accuracy)

        results.append({
            "question": question,
            "predicted": predicted,
            "actual": actual,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "responses": responses,
        })

    return results


def compute_metrics(results):
    """Compute evaluation metrics."""
    total = len(results)

    # Exact match (predicted == actual)
    exact_match = sum(1 for r in results if r["predicted"] == r["actual"])

    # Confusion matrix for can/cannot (treating uncertain as cannot for simplicity)
    def simplify(ability):
        return "can" if ability == "can" else "cannot"

    tp = sum(1 for r in results if simplify(r["predicted"]) == "can" and simplify(r["actual"]) == "can")
    tn = sum(1 for r in results if simplify(r["predicted"]) == "cannot" and simplify(r["actual"]) == "cannot")
    fp = sum(1 for r in results if simplify(r["predicted"]) == "can" and simplify(r["actual"]) == "cannot")  # Overconfident
    fn = sum(1 for r in results if simplify(r["predicted"]) == "cannot" and simplify(r["actual"]) == "can")  # Underconfident

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total": total,
        "exact_match": exact_match,
        "exact_match_rate": exact_match / total,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,  # Overconfident
        "false_negative": fn,  # Underconfident
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--lora_path", type=str, default=str(project_root / "outputs/metacog"))
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--num_trials", type=int, default=5, help="Trials per question for actual ability")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    print(f"Loading TriviaQA {args.split} split...")
    samples = load_triviaqa(split=args.split, num_samples=args.num_samples)
    print(f"Loaded {len(samples)} samples")

    print(f"Loading trained model from {args.lora_path}")
    model, tokenizer = load_trained_model(args.model, args.lora_path)

    print(f"Evaluating metacognition ({args.num_trials} trials per question)...")
    results = evaluate_metacognition(model, tokenizer, samples, args.num_trials)

    metrics = compute_metrics(results)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples: {metrics['total']}")
    print(f"\nExact match (predicted == actual): {metrics['exact_match']} ({metrics['exact_match_rate']*100:.1f}%)")
    print(f"\nConfusion Matrix (can vs cannot):")
    print(f"  True Positive (correctly confident):  {metrics['true_positive']}")
    print(f"  True Negative (correctly uncertain):  {metrics['true_negative']}")
    print(f"  False Positive (overconfident):       {metrics['false_positive']}")
    print(f"  False Negative (underconfident):      {metrics['false_negative']}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.1f}%")
    print(f"  Precision: {metrics['precision']*100:.1f}%")
    print(f"  Recall:    {metrics['recall']*100:.1f}%")
    print(f"  F1 Score:  {metrics['f1']*100:.1f}%")

    # Show some examples
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)

    # Show some correct predictions
    correct = [r for r in results if r["predicted"] == r["actual"]][:3]
    if correct:
        print("\n--- Correct Predictions ---")
        for r in correct:
            print(f"Q: {r['question'][:80]}...")
            print(f"   Predicted: {r['predicted']}, Actual: {r['actual']} (acc: {r['accuracy']*100:.0f}%)")

    # Show some wrong predictions
    wrong = [r for r in results if r["predicted"] != r["actual"]][:3]
    if wrong:
        print("\n--- Wrong Predictions ---")
        for r in wrong:
            print(f"Q: {r['question'][:80]}...")
            print(f"   Predicted: {r['predicted']}, Actual: {r['actual']} (acc: {r['accuracy']*100:.0f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
