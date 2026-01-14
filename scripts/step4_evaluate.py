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
    def __init__(self, base_model: str, lora_path: str = None, inference_batch_size: int = 16):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Important for batch generation

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # Support evaluation without LoRA (baseline)
        if lora_path and lora_path.lower() != "none":
            self.model = PeftModel.from_pretrained(base, lora_path)
        else:
            self.model = base
        self.model.eval()
        self.inference_batch_size = inference_batch_size

    def predict_ability(self, question: str) -> str:
        """Ask trained model to assess its ability to answer."""
        # Use chat template for instruction-tuned models
        messages = [
            {"role": "system", "content": "Assess whether you can answer the question correctly. Output your judgment in \\boxed{}, using ONLY one word: yes (definitely correct), uncertain (likely correct but not sure), or no (unlikely correct)."},
            {"role": "user", "content": f"Can you answer this question correctly?\n\nQuestion: {question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

        # Parse \boxed{} format
        import re
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

    def answer_questions_batch(self, questions: list, num_trials: int) -> list:
        """
        Get multiple answers for multiple questions using true batch generation.

        Args:
            questions: List of questions to answer
            num_trials: Number of responses per question

        Returns:
            List of lists, each containing num_trials responses for each question
        """
        # Create prompts for all questions using chat template, repeated num_trials times
        prompts = []
        for question in questions:
            messages = [
                {"role": "system", "content": "Answer the question concisely and directly."},
                {"role": "user", "content": question}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.extend([prompt] * num_trials)

        # Batch tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=1.0,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode all responses
        all_responses = []
        for i, output in enumerate(outputs):
            input_len = (inputs["attention_mask"][i] == 1).sum().item()
            response = self.tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True
            ).strip()
            all_responses.append(response)

        # Group responses by question
        results = []
        for i in range(len(questions)):
            start_idx = i * num_trials
            end_idx = start_idx + num_trials
            results.append(all_responses[start_idx:end_idx])

        return results

    def evaluate(self, samples, num_trials: int = 5):
        """Evaluate metacognition accuracy with batch inference."""
        results = []

        # Process samples in batches
        for batch_start in tqdm(range(0, len(samples), self.inference_batch_size), desc="Evaluating trained model"):
            batch_end = min(batch_start + self.inference_batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]

            # 1. Get predicted abilities for batch (still one at a time for simplicity)
            predicted_abilities = []
            for sample in batch_samples:
                predicted = self.predict_ability(sample["question"])
                predicted_abilities.append(predicted)

            # 2. Batch generate responses for all questions in batch
            questions = [s["question"] for s in batch_samples]
            all_responses = self.answer_questions_batch(questions, num_trials)

            # 3. Evaluate each sample
            for i, sample in enumerate(batch_samples):
                gold_answers = sample.get("normalized_answers", sample["answers"])
                responses = all_responses[i]
                correct_count = sum(1 for r in responses if is_correct(r, gold_answers))

                accuracy = correct_count / num_trials
                actual = classify_ability(accuracy)

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

    return {
        "total": total,
        "exact_match": exact_match,
        "exact_match_rate": exact_match / total if total > 0 else 0,
        "confusion": confusion,
        "pred_counts": pred_counts,
        "actual_counts": actual_counts,
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
    print("EVALUATION RESULTS (After Training)")
    print("=" * 60)
    print(f"Total samples: {metrics['total']}")
    print(f"\nExact match (predicted == actual): {metrics['exact_match']} ({metrics['exact_match_rate']*100:.1f}%)")

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
