"""
HotpotQA Binary Classification Training Pipeline.

This script:
1. Loads HotpotQA (distractor config)
2. Collects responses to determine can/cannot labels
3. Trains binary judgment model
4. Evaluates with AUROC

Usage:
    python scripts/run_hotpotqa_binary.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --train_samples 5000 \
        --val_samples 1000 \
        --output_dir experiments/hotpotqa_binary
"""

import argparse
import os
import sys
import json
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm

from src.hotpotqa_loader import load_hotpotqa, format_hotpotqa_prompt, extract_answer_from_response
from src.hotpotqa_label_generator import build_training_dataset, SYSTEM_PROMPT
from src.evaluator import normalize_answer, is_correct
from src.dataset_builder import save_to_jsonl, load_from_jsonl


def collect_responses_and_labels(
    samples,
    model,
    tokenizer,
    num_trials: int = 1,
    batch_size: int = 8,
    max_new_tokens: int = 128,
):
    """
    Collect responses and determine binary labels (can/cannot).

    For binary mode: temperature=0, single trial.
    If correct -> can, else -> cannot.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = []

    # Process in batches
    for i in tqdm(range(0, len(samples), batch_size), desc="Collecting responses"):
        batch = samples[i:i + batch_size]
        prompts = [format_hotpotqa_prompt(s) for s in batch]

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        # Generate (greedy for binary)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode responses
        for j, (sample, output) in enumerate(zip(batch, outputs)):
            input_len = inputs["attention_mask"][j].sum().item()
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()

            # Extract answer
            extracted_answer = extract_answer_from_response(response)

            # Check correctness
            correct = is_correct(extracted_answer, sample["answers"])
            ability = "can" if correct else "cannot"

            result = sample.copy()
            result["response"] = response
            result["extracted_answer"] = extracted_answer
            result["correct"] = correct
            result["ability"] = ability
            results.append(result)

    return results


def evaluate_with_auroc(
    samples,
    model,
    tokenizer,
    batch_size: int = 8,
    max_new_tokens: int = 128,
):
    """
    Evaluate model and compute AUROC.

    Returns judgment accuracy and AUROC (based on P(yes) as confidence).
    """
    results = []
    all_probs = []
    all_labels = []
    all_preds = []

    # Get token IDs for "yes" and "no"
    yes_token = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_token = tokenizer.encode("no", add_special_tokens=False)[0]

    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch = samples[i:i + batch_size]

        # Build judgment prompts
        prompts = []
        for s in batch:
            user_content = f"""Context:
{s['context']}

Question: {s['question']}

Can you answer this question correctly based on the context? Provide your judgment and answer."""

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Get first token logits for P(yes) vs P(no)
        first_token_logits = outputs.scores[0]  # [batch, vocab]

        for j, sample in enumerate(batch):
            # Get P(yes) and P(no) from first generated token
            logits = first_token_logits[j]
            probs = torch.softmax(logits, dim=-1)

            p_yes = probs[yes_token].item()
            p_no = probs[no_token].item()

            # Normalize to get confidence
            confidence = p_yes / (p_yes + p_no + 1e-10)

            # Decode full response
            input_len = inputs["attention_mask"][j].sum().item()
            response = tokenizer.decode(outputs.sequences[j][input_len:], skip_special_tokens=True).strip()

            # Parse judgment from response
            judgment_match = re.search(r"\\boxed\{(\w+)\}", response)
            if judgment_match:
                judgment = judgment_match.group(1).lower()
                predicted_can = judgment == "yes"
            else:
                # Fallback: check if response starts with yes/no
                predicted_can = response.lower().startswith("yes") or "boxed{yes}" in response.lower()

            # Extract answer
            answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            extracted_answer = answer_match.group(1).strip() if answer_match else response

            # Check actual correctness
            actual_correct = is_correct(extracted_answer, sample["answers"])

            # Ground truth label (based on what model actually got right/wrong)
            # For AUROC: we want confidence to correlate with actual correctness
            all_probs.append(confidence)
            all_labels.append(1 if actual_correct else 0)
            all_preds.append(1 if predicted_can else 0)

            results.append({
                "id": sample["id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "response": response,
                "extracted_answer": extracted_answer,
                "predicted_can": predicted_can,
                "actual_correct": actual_correct,
                "confidence": confidence,
                "p_yes": p_yes,
                "p_no": p_no,
            })

    # Compute metrics
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    judgment_acc = accuracy_score(all_labels, all_preds)

    # Detailed report
    report = classification_report(all_labels, all_preds, target_names=["cannot", "can"], output_dict=True)

    metrics = {
        "auroc": auroc,
        "judgment_accuracy": judgment_acc,
        "total_samples": len(results),
        "actual_correct_count": sum(all_labels),
        "predicted_can_count": sum(all_preds),
        "classification_report": report,
    }

    return results, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="experiments/hotpotqa_binary")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--skip_training", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.checkpoint if args.checkpoint else args.model
    print(f"\nLoading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load data
    print(f"\nLoading HotpotQA...")
    train_data = load_hotpotqa("train", num_samples=args.train_samples)
    val_data = load_hotpotqa("validation", num_samples=args.val_samples)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    if not args.skip_training:
        # Step 1: Collect responses and labels
        print("\n" + "="*60)
        print("Step 1: Collecting responses to determine labels...")
        print("="*60)

        responses_file = output_dir / "train_responses.jsonl"
        if responses_file.exists():
            print(f"Loading cached responses from {responses_file}")
            labeled_train = load_from_jsonl(str(responses_file))
        else:
            labeled_train = collect_responses_and_labels(
                train_data, model, tokenizer, batch_size=args.batch_size
            )
            save_to_jsonl(labeled_train, str(responses_file))

        # Stats
        can_count = sum(1 for s in labeled_train if s["ability"] == "can")
        cannot_count = len(labeled_train) - can_count
        print(f"Labels: can={can_count}, cannot={cannot_count}")

        # Step 2: Build training dataset
        print("\n" + "="*60)
        print("Step 2: Building training dataset...")
        print("="*60)

        training_data = build_training_dataset(labeled_train)
        training_file = output_dir / "training_data.jsonl"
        save_to_jsonl(training_data, str(training_file))
        print(f"Saved {len(training_data)} training samples to {training_file}")

        # Step 3: Train
        print("\n" + "="*60)
        print("Step 3: Training...")
        print("="*60)

        # Use the existing trainer
        from src.trainer import train_judgment_model

        checkpoint_dir = output_dir / "checkpoint"
        train_judgment_model(
            model_name=args.model,
            training_data=training_data,
            output_dir=str(checkpoint_dir),
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=4,
            gradient_accumulation_steps=4,
        )

        # Reload trained model
        print("\nReloading trained model...")
        del model
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

    # Step 4: Evaluate
    print("\n" + "="*60)
    print("Step 4: Evaluating with AUROC...")
    print("="*60)

    eval_results, metrics = evaluate_with_auroc(
        val_data, model, tokenizer, batch_size=args.batch_size
    )

    # Save results
    save_to_jsonl(eval_results, str(output_dir / "eval_results.jsonl"))

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print results
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"Judgment Accuracy: {metrics['judgment_accuracy']:.4f}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Actual correct: {metrics['actual_correct_count']}")
    print(f"Predicted can: {metrics['predicted_can_count']}")

    # Save summary
    summary = f"""HotpotQA Binary Classification Results
========================================
Model: {args.model}
Train samples: {args.train_samples}
Val samples: {args.val_samples}

AUROC: {metrics['auroc']:.4f}
Judgment Accuracy: {metrics['judgment_accuracy']:.4f}

Total samples: {metrics['total_samples']}
Actual correct: {metrics['actual_correct_count']}
Predicted can: {metrics['predicted_can_count']}
"""

    with open(output_dir / "results.txt", "w") as f:
        f.write(summary)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
