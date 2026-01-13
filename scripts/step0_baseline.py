"""
Step 0: Baseline evaluation - test original model's metacognition ability.

Same as step 4 but uses the base model without LoRA fine-tuning.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.data_loader import load_triviaqa
from src.evaluator import is_correct, classify_ability


def load_base_model(model_name: str):
    """Load base model without LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    return model, tokenizer


def predict_ability(model, tokenizer, question: str) -> str:
    """Ask model to assess its ability to answer."""
    prompt = f"Before answering, assess your ability to answer this question:\n{question}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.1,
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


def evaluate_metacognition(model, tokenizer, samples, num_trials: int = 5):
    """Evaluate metacognition accuracy."""
    results = []

    for sample in tqdm(samples, desc="Evaluating baseline"):
        question = sample["question"]
        gold_answers = sample.get("normalized_answers", sample["answers"])

        # 1. Get predicted ability
        predicted = predict_ability(model, tokenizer, question)

        # 2. Test actual ability (answer N times)
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
        })

    return results


def compute_metrics(results):
    """Compute evaluation metrics."""
    total = len(results)

    exact_match = sum(1 for r in results if r["predicted"] == r["actual"])

    def simplify(ability):
        return "can" if ability == "can" else "cannot"

    tp = sum(1 for r in results if simplify(r["predicted"]) == "can" and simplify(r["actual"]) == "can")
    tn = sum(1 for r in results if simplify(r["predicted"]) == "cannot" and simplify(r["actual"]) == "cannot")
    fp = sum(1 for r in results if simplify(r["predicted"]) == "can" and simplify(r["actual"]) == "cannot")
    fn = sum(1 for r in results if simplify(r["predicted"]) == "cannot" and simplify(r["actual"]) == "can")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total": total,
        "exact_match": exact_match,
        "exact_match_rate": exact_match / total if total > 0 else 0,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / total if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    print(f"Loading TriviaQA {args.split} split...")
    samples = load_triviaqa(split=args.split, num_samples=args.num_samples)
    print(f"Loaded {len(samples)} samples")

    print(f"Loading base model: {args.model}")
    model, tokenizer = load_base_model(args.model)

    print(f"Evaluating baseline metacognition ({args.num_trials} trials per question)...")
    results = evaluate_metacognition(model, tokenizer, samples, args.num_trials)

    metrics = compute_metrics(results)

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS (Before Training)")
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

    print("\nDone!")


if __name__ == "__main__":
    main()
