"""
Step 4: Evaluate trained metacognition model on test set.

Compares model's self-assessment (can/uncertain/cannot) with actual performance.
Supports batch inference for better GPU utilization.
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


class TrainedModelEvaluator:
    def __init__(self, base_model: str, lora_path: str, inference_batch_size: int = 16):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Important for batch generation

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base, lora_path)
        self.model.eval()
        self.inference_batch_size = inference_batch_size

    def predict_ability(self, question: str) -> str:
        """Ask trained model to assess its ability to answer."""
        prompt = f"Before answering, assess your ability to answer this question:\n{question}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.1,  # Low temperature for consistent prediction
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
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

    def answer_question_batch(self, question: str, num_trials: int) -> list:
        """Get multiple answers for a question using batch generation."""
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=num_trials,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        responses = []
        input_len = inputs["input_ids"].shape[1]
        for i in range(num_trials):
            response = self.tokenizer.decode(
                outputs[i][input_len:],
                skip_special_tokens=True
            ).strip()
            responses.append(response)

        return responses

    def evaluate(self, samples, num_trials: int = 5):
        """Evaluate metacognition accuracy with batch inference."""
        results = []

        for sample in tqdm(samples, desc="Evaluating trained model"):
            question = sample["question"]
            gold_answers = sample.get("normalized_answers", sample["answers"])

            # 1. Get predicted ability
            predicted = self.predict_ability(question)

            # 2. Test actual ability (batch generate num_trials responses)
            responses = self.answer_question_batch(question, num_trials)
            correct_count = sum(1 for r in responses if is_correct(r, gold_answers))

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
    parser.add_argument("--inference_batch_size", type=int, default=16, help="Batch size for inference")
    args = parser.parse_args()

    print(f"Loading TriviaQA {args.split} split...")
    samples = load_triviaqa(split=args.split, num_samples=args.num_samples)
    print(f"Loaded {len(samples)} samples")

    print(f"Loading trained model from {args.lora_path}")
    print(f"Inference batch size: {args.inference_batch_size}")
    evaluator = TrainedModelEvaluator(args.model, args.lora_path, args.inference_batch_size)

    print(f"Evaluating metacognition ({args.num_trials} trials per question)...")
    results = evaluator.evaluate(samples, args.num_trials)

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
