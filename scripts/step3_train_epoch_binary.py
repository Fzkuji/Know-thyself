"""
Step 3 Binary: Train model for one epoch (binary judgment training).

Binary version:
- Only "can" and "cannot" (no "uncertain")
- Temperature=0 for all inference (greedy decoding)
- Single trial per question (correct or not)

This script trains the model for exactly ONE epoch, then exits.
Use this in a shell loop to interleave training and evaluation.
"""

import argparse
import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset_builder import load_from_jsonl
from src.label_generator import SYSTEM_PROMPT
from src.evaluator import is_correct


def classify_ability_binary(correct: bool) -> str:
    """Binary classification: correct -> can, incorrect -> cannot."""
    return "can" if correct else "cannot"


def generate_qa_response_greedy(model, tokenizer, question: str) -> str:
    """Generate a single QA response with greedy decoding (temperature=0)."""
    messages = [
        {"role": "system", "content": "Answer the question directly and concisely."},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    ).to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=None,  # Greedy decoding
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    ).strip()

    return response


def compute_realtime_ability_binary(model, tokenizer, question: str, gold_answers: list) -> str:
    """Compute model's current ability with single greedy trial."""
    response = generate_qa_response_greedy(model, tokenizer, question)
    correct = is_correct(response, gold_answers)
    return classify_ability_binary(correct)


def test_judgment_binary(model, tokenizer, question: str, expected_ability: str, system_prompt: str) -> bool:
    """Test if model predicts the correct binary ability (greedy decoding)."""
    import re

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Can you answer this question correctly?\n\nQuestion: {question}"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=None,  # Greedy decoding
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip().lower()

    # Parse prediction (binary: yes -> can, no -> cannot)
    match = re.search(r'\\boxed\{(\w+)\}', response)
    if match:
        predicted = match.group(1).lower()
    elif "yes" in response:
        predicted = "yes"
    else:
        predicted = "no"

    # Map to binary ability
    predicted_ability = "can" if predicted == "yes" else "cannot"

    return predicted_ability == expected_ability


def build_judgment_sample_binary(question: str, ability: str, system_prompt: str) -> dict:
    """Build a training sample for binary judgment prediction."""
    # Binary: can -> Yes, cannot -> No
    answer = "Yes" if ability == "can" else "No"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Can you answer this question correctly?\n\nQuestion: {question}"},
        {"role": "assistant", "content": f"\\boxed{{{answer}}}"}
    ]

    return {"messages": messages, "ability": ability}


def train_one_epoch_binary(
    model,
    tokenizer,
    samples: list,
    system_prompt: str,
    optimizer,
    skip_correct: bool = True,
    use_realtime_labels: bool = True,
) -> dict:
    """Train for one epoch with binary classification."""
    epoch_stats = {
        "total": len(samples),
        "skipped": 0,
        "trained": 0,
        "by_ability_trained": {"can": 0, "cannot": 0},
        "losses": [],
    }

    pbar = tqdm(samples, desc="Training (binary)")

    for sample in pbar:
        question = sample.get("question", "")
        gold_answers = sample.get("normalized_answers", sample.get("answers", []))

        # Compute binary ability (single greedy trial)
        if use_realtime_labels and question and gold_answers:
            ability = compute_realtime_ability_binary(model, tokenizer, question, gold_answers)
        else:
            # Convert existing label to binary
            orig_ability = sample.get("ability", "")
            if orig_ability == "can":
                ability = "can"
            else:
                ability = "cannot"  # uncertain and cannot both map to cannot

        # Check if already correct (skip training for this sample)
        if skip_correct and question and ability and test_judgment_binary(model, tokenizer, question, ability, system_prompt):
            epoch_stats["skipped"] += 1
            pbar.set_postfix({"skip": epoch_stats["skipped"], "train": epoch_stats["trained"]})
            continue

        # Build training sample
        if use_realtime_labels:
            train_sample = build_judgment_sample_binary(question, ability, system_prompt)
        else:
            # Convert existing sample to binary format
            train_sample = build_judgment_sample_binary(question, ability, system_prompt)

        # Tokenize
        text = tokenizer.apply_chat_template(
            train_sample["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs["labels"] = inputs["input_ids"].clone()
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Train step
        model.train()
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        epoch_stats["losses"].append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_stats["trained"] += 1
        if ability:
            epoch_stats["by_ability_trained"][ability] += 1

        pbar.set_postfix({
            "skip": epoch_stats["skipped"],
            "train": epoch_stats["trained"],
            "loss": f"{loss.item():.4f}",
        })

    if epoch_stats["losses"]:
        epoch_stats["avg_loss"] = sum(epoch_stats["losses"]) / len(epoch_stats["losses"])

    return epoch_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model path (base model or continued from previous epoch)")
    parser.add_argument("--input", type=str, required=True,
                        help="Training data JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory to save the model")
    parser.add_argument("--epoch", type=int, default=1,
                        help="Current epoch number (for logging)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--skip_correct", action="store_true", default=True,
                        help="Skip samples where model already judges correctly")
    parser.add_argument("--use_realtime_labels", action="store_true", default=True,
                        help="Compute ability labels in real-time (single greedy trial)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Epoch {args.epoch}: Binary Single-epoch training")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Training data: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Learning rate: {args.lr}")
    print(f"Mode: Binary (can/cannot), temperature=0, single trial")
    print(f"Skip correct: {args.skip_correct}")
    print(f"Real-time labels: {args.use_realtime_labels}")

    # Load training data
    print(f"\nLoading training data...")
    training_data = load_from_jsonl(args.input)
    print(f"Loaded {len(training_data)} samples")

    # Load model (device_map="auto" for multi-GPU support)
    print(f"\nLoading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Train one epoch
    print(f"\nTraining epoch {args.epoch} (binary mode)...")
    stats = train_one_epoch_binary(
        model=model,
        tokenizer=tokenizer,
        samples=training_data,
        system_prompt=SYSTEM_PROMPT,
        optimizer=optimizer,
        skip_correct=args.skip_correct,
        use_realtime_labels=args.use_realtime_labels,
    )

    # Print summary
    print(f"\n  Epoch {args.epoch} Summary (Binary):")
    print(f"    Trained: {stats['trained']}, Skipped: {stats['skipped']}")
    if stats.get("avg_loss"):
        print(f"    Average loss: {stats['avg_loss']:.4f}")
    print(f"    By ability: can={stats['by_ability_trained']['can']}, "
          f"cannot={stats['by_ability_trained']['cannot']}")

    # Save model
    print(f"\nSaving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save stats
    stats_file = Path(args.output_dir) / f"epoch_{args.epoch}_stats.json"
    with open(stats_file, "w") as f:
        # Convert losses list to just avg_loss for JSON
        save_stats = {k: v for k, v in stats.items() if k != "losses"}
        json.dump(save_stats, f, indent=2)
    print(f"Stats saved to {stats_file}")

    print(f"\nEpoch {args.epoch} complete!")


if __name__ == "__main__":
    main()
