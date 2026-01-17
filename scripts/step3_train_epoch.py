"""
Step 3: Train model for one epoch (judgment training).

This script trains the model for exactly ONE epoch, then exits.
Use this in a shell loop to interleave training and evaluation:

    for epoch in 1 2 3; do
        python step3_train_epoch.py --model $MODEL --epoch $epoch ...
        python step4_evaluate.py --model $OUTPUT ...
    done

Supports:
- Real-time label generation during training
- Skip samples where model already predicts correctly
- Continue from existing model (for multi-epoch training)
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
from src.evaluator import is_correct, classify_ability


def generate_qa_responses(model, tokenizer, question: str, num_trials: int):
    """Generate multiple QA responses for a question (batched)."""
    messages = [
        {"role": "system", "content": "Answer the question directly and concisely."},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Batch: repeat the same prompt num_trials times
    prompts = [prompt] * num_trials
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    responses = []
    input_len = inputs["input_ids"].shape[1]
    for output in outputs:
        response = tokenizer.decode(
            output[input_len:],
            skip_special_tokens=True
        ).strip()
        responses.append(response)

    return responses


def compute_realtime_ability(model, tokenizer, question: str, gold_answers: list, num_trials: int) -> str:
    """Compute model's current ability by generating QA responses."""
    responses = generate_qa_responses(model, tokenizer, question, num_trials)
    correct_count = sum(1 for r in responses if is_correct(r, gold_answers))
    return classify_ability(correct_count, len(responses))


def test_judgment(model, tokenizer, question: str, expected_ability: str, system_prompt: str) -> bool:
    """Test if model predicts the correct ability."""
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
            temperature=0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip().lower()

    # Parse prediction
    match = re.search(r'\\boxed\{(\w+)\}', response)
    if match:
        predicted = match.group(1).lower()
    elif "yes" in response:
        predicted = "yes"
    elif "uncertain" in response:
        predicted = "uncertain"
    else:
        predicted = "no"

    ability_map = {"yes": "can", "uncertain": "uncertain", "no": "cannot"}
    predicted_ability = ability_map.get(predicted, "cannot")

    return predicted_ability == expected_ability


def build_judgment_sample(question: str, ability: str, system_prompt: str) -> dict:
    """Build a training sample for judgment prediction."""
    ability_to_answer = {
        "can": "Yes",
        "uncertain": "Uncertain",
        "cannot": "No"
    }
    answer = ability_to_answer.get(ability, "No")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Can you answer this question correctly?\n\nQuestion: {question}"},
        {"role": "assistant", "content": f"\\boxed{{{answer}}}"}
    ]

    return {"messages": messages, "ability": ability}


def train_one_epoch(
    model,
    tokenizer,
    samples: list,
    system_prompt: str,
    optimizer,
    num_trials: int = 10,
    skip_correct: bool = True,
    use_realtime_labels: bool = True,
) -> dict:
    """Train for one epoch."""
    epoch_stats = {
        "total": len(samples),
        "skipped": 0,
        "trained": 0,
        "by_ability_trained": {"can": 0, "uncertain": 0, "cannot": 0},
        "losses": [],
    }

    pbar = tqdm(samples, desc="Training")

    for sample in pbar:
        question = sample.get("question", "")
        gold_answers = sample.get("normalized_answers", sample.get("answers", []))

        # Compute ability
        if use_realtime_labels and question and gold_answers:
            ability = compute_realtime_ability(model, tokenizer, question, gold_answers, num_trials)
        else:
            ability = sample.get("ability", "")

        # Check if already correct (skip training for this sample)
        if skip_correct and question and ability and test_judgment(model, tokenizer, question, ability, system_prompt):
            epoch_stats["skipped"] += 1
            pbar.set_postfix({"skip": epoch_stats["skipped"], "train": epoch_stats["trained"]})
            continue

        # Build training sample
        if use_realtime_labels:
            train_sample = build_judgment_sample(question, ability, system_prompt)
        else:
            train_sample = sample

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
    parser.add_argument("--num_trials", type=int, default=10,
                        help="Number of QA trials for real-time ability assessment")
    parser.add_argument("--skip_correct", action="store_true", default=True,
                        help="Skip samples where model already judges correctly")
    parser.add_argument("--use_realtime_labels", action="store_true", default=True,
                        help="Compute ability labels in real-time")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Epoch {args.epoch}: Single-epoch training")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Training data: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Learning rate: {args.lr}")
    print(f"QA trials: {args.num_trials}")
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
    print(f"\nTraining epoch {args.epoch}...")
    stats = train_one_epoch(
        model=model,
        tokenizer=tokenizer,
        samples=training_data,
        system_prompt=SYSTEM_PROMPT,
        optimizer=optimizer,
        num_trials=args.num_trials,
        skip_correct=args.skip_correct,
        use_realtime_labels=args.use_realtime_labels,
    )

    # Print summary
    print(f"\n  Epoch {args.epoch} Summary:")
    print(f"    Trained: {stats['trained']}, Skipped: {stats['skipped']}")
    if stats.get("avg_loss"):
        print(f"    Average loss: {stats['avg_loss']:.4f}")
    print(f"    By ability: can={stats['by_ability_trained']['can']}, "
          f"uncertain={stats['by_ability_trained']['uncertain']}, "
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
