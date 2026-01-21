"""
Unified Judgment Training with DeepSpeed ZeRO-3.

Supports:
- Label mode: binary (temp=0, single trial) or uncertainty (temp>0, N trials)
- Training mode: batch (test all -> train failed) or adaptive (test each -> train each)

Usage:
    # Binary + Batch mode (default)
    deepspeed --num_gpus=8 scripts/train_judgment_deepspeed.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output_dir experiments/judgment \
        --deepspeed configs/ds_config_zero3.json

    # Uncertainty + Adaptive mode
    deepspeed --num_gpus=8 scripts/train_judgment_deepspeed.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output_dir experiments/judgment \
        --label_mode uncertainty \
        --training_mode adaptive \
        --num_trials 10 \
        --deepspeed configs/ds_config_zero3.json
"""

import argparse
import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from dataclasses import dataclass
from typing import Any, Dict, List
from tqdm import tqdm

from src.data_loader import load_triviaqa, format_question_prompt
from src.dataset_builder import load_from_jsonl, save_to_jsonl
from src.evaluator import is_correct, classify_ability, classify_ability_binary
from src.label_generator import SYSTEM_PROMPT, build_training_sample


@dataclass
class DataCollatorForCausalLM:
    """
    Data collator for causal language modeling that properly pads both
    input_ids and labels to the same length within a batch.
    """
    tokenizer: Any
    padding: bool = True
    max_length: int = None
    pad_to_multiple_of: int = None
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [feature.pop("labels") for feature in features] if "labels" in features[0] else None

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padded_labels = []
            for label in labels:
                remainder = max_label_length - len(label)
                padded_label = list(label) + [self.label_pad_token_id] * remainder
                padded_labels.append(padded_label)

            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


def collect_responses_binary(samples, model, tokenizer, batch_size=8, max_new_tokens=64,
                             local_rank=0, world_size=1, show_progress=True):
    """
    Collect responses with temperature=0, single trial.
    Returns samples with 'ability' field (can/cannot).

    Supports distributed: each rank processes its shard of data.
    """
    # Shard data across ranks
    shard_size = (len(samples) + world_size - 1) // world_size
    start_idx = local_rank * shard_size
    end_idx = min(start_idx + shard_size, len(samples))
    local_samples = samples[start_idx:end_idx]

    results = []
    model.eval()

    iterator = range(0, len(local_samples), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Rank {local_rank} collecting responses")

    for i in iterator:
        batch = local_samples[i:i + batch_size]
        prompts = [format_question_prompt(s["question"]) for s in batch]

        # Apply chat template
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted)

        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, (sample, output) in enumerate(zip(batch, outputs)):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()

            gold_answers = sample.get("normalized_answers", sample.get("answers", []))
            correct = is_correct(response, gold_answers)
            ability = classify_ability_binary(correct)

            result = sample.copy()
            result["response"] = response
            result["correct"] = correct
            result["ability"] = ability
            result["_original_idx"] = start_idx + i + j  # Track original order
            results.append(result)

    return results


def collect_responses_uncertainty(samples, model, tokenizer, num_trials=10, batch_size=8, max_new_tokens=64, temperature=0.7):
    """
    Collect responses with temperature>0, multiple trials.
    Returns samples with 'ability' field (can/uncertain/cannot).
    """
    results = []
    model.eval()

    for sample in tqdm(samples, desc="Collecting responses (uncertainty)"):
        prompt = format_question_prompt(sample["question"])
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(
            [formatted] * num_trials,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )

        responses = []
        correct_count = 0
        gold_answers = sample.get("normalized_answers", sample.get("answers", []))

        for k, output in enumerate(outputs):
            input_len = inputs["input_ids"][k].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            responses.append(response)
            if is_correct(response, gold_answers):
                correct_count += 1

        ability = classify_ability(correct_count, num_trials)

        result = sample.copy()
        result["responses"] = responses
        result["correct_count"] = correct_count
        result["ability"] = ability
        results.append(result)

    return results


def preprocess_function(examples, tokenizer):
    """Preprocess judgment samples for training."""
    texts = []
    for i in range(len(examples["question"])):
        question = examples["question"][i]
        ability = examples["ability"][i]

        sample = build_training_sample(question, ability)
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding=False,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def test_judgment_accuracy(samples, model, tokenizer, batch_size=8):
    """
    Test model's judgment accuracy.
    Returns list of samples with 'judgment_correct' field.
    """
    results = []
    model.eval()

    for i in tqdm(range(0, len(samples), batch_size), desc="Testing judgments"):
        batch = samples[i:i + batch_size]

        # Build judgment prompts
        prompts = []
        for s in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Can you answer this question correctly?\n\nQuestion: {s['question']}"},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, (sample, output) in enumerate(zip(batch, outputs)):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip().lower()

            # Parse judgment
            if "\\boxed{yes}" in response or "boxed{yes}" in response:
                predicted = "can"
            elif "\\boxed{uncertain}" in response or "boxed{uncertain}" in response:
                predicted = "uncertain"
            else:
                predicted = "cannot"

            ground_truth = sample["ability"]
            judgment_correct = (predicted == ground_truth)

            result = sample.copy()
            result["predicted_judgment"] = predicted
            result["judgment_correct"] = judgment_correct
            results.append(result)

    return results


def run_batch_training(args, model, tokenizer, samples, epoch, is_main):
    """
    Batch training mode: test all samples, then train on incorrect judgments.
    """
    if is_main:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}: Testing all samples...")
        print(f"{'='*60}")

    # Test judgments
    tested_samples = test_judgment_accuracy(samples, model, tokenizer, batch_size=args.batch_size)

    # Filter incorrect judgments
    incorrect_samples = [s for s in tested_samples if not s["judgment_correct"]]
    correct_count = len(samples) - len(incorrect_samples)

    if is_main:
        print(f"Judgment accuracy: {correct_count}/{len(samples)} ({correct_count/len(samples)*100:.1f}%)")
        print(f"Samples to train: {len(incorrect_samples)}")

    if len(incorrect_samples) == 0:
        if is_main:
            print("All judgments correct! No training needed.")
        return tested_samples, 0.0

    # Build training dataset from incorrect samples
    dataset_dict = {
        "question": [s["question"] for s in incorrect_samples],
        "ability": [s["ability"] for s in incorrect_samples],
    }
    dataset = Dataset.from_dict(dataset_dict)

    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Training with automatic OOM recovery
    current_batch_size = args.batch_size
    current_grad_accum = args.gradient_accumulation_steps
    effective_batch_size = current_batch_size * current_grad_accum

    while current_batch_size >= 1:
        try:
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=current_batch_size,
                gradient_accumulation_steps=current_grad_accum,
                learning_rate=args.lr,
                weight_decay=0.01,
                warmup_ratio=0.1,
                logging_steps=10,
                save_strategy="no",
                bf16=True,
                deepspeed=args.deepspeed,
                gradient_checkpointing=True,
                report_to="none",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            )

            data_collator = DataCollatorForCausalLM(
                tokenizer=tokenizer,
                padding=True,
                label_pad_token_id=-100,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                processing_class=tokenizer,
            )

            if is_main:
                print(f"\nTraining on {len(incorrect_samples)} samples (batch_size={current_batch_size}, grad_accum={current_grad_accum})...")

            train_result = trainer.train()
            return tested_samples, train_result.training_loss

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                # Clear memory
                torch.cuda.empty_cache()

                # Halve batch size, double gradient accumulation to maintain effective batch size
                new_batch_size = max(1, current_batch_size // 2)
                if new_batch_size == current_batch_size:
                    # Already at batch_size=1, cannot reduce further
                    raise RuntimeError(f"OOM even with batch_size=1. Try gradient_checkpointing or CPU offload.") from e

                current_grad_accum = current_grad_accum * (current_batch_size // new_batch_size)
                current_batch_size = new_batch_size

                if is_main:
                    print(f"\n⚠️ OOM detected! Reducing batch_size to {current_batch_size}, grad_accum to {current_grad_accum}")
            else:
                raise

    raise RuntimeError("Training failed: could not find working batch size")


def run_adaptive_training(args, model, tokenizer, samples, epoch, is_main):
    """
    Adaptive training mode: test each sample, train immediately if incorrect.
    """
    if is_main:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}: Adaptive training...")
        print(f"{'='*60}")

    correct_count = 0
    trained_count = 0
    total_loss = 0.0

    for idx, sample in enumerate(tqdm(samples, desc="Adaptive training")):
        # Test single sample
        tested = test_judgment_accuracy([sample], model, tokenizer, batch_size=1)[0]

        if tested["judgment_correct"]:
            correct_count += 1
            continue

        # Train on this sample
        dataset_dict = {
            "question": [sample["question"]],
            "ability": [sample["ability"]],
        }
        dataset = Dataset.from_dict(dataset_dict)

        tokenized_dataset = dataset.map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
        )

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.0,
            logging_steps=1,
            save_strategy="no",
            bf16=True,
            deepspeed=args.deepspeed,
            gradient_checkpointing=True,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )

        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
        )

        train_result = trainer.train()
        trained_count += 1
        total_loss += train_result.training_loss

    if is_main:
        print(f"\nCorrect judgments: {correct_count}/{len(samples)} ({correct_count/len(samples)*100:.1f}%)")
        print(f"Trained samples: {trained_count}")

    avg_loss = total_loss / trained_count if trained_count > 0 else 0.0
    return None, avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed config file")

    # Data options
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSONL file with samples (if not provided, loads TriviaQA)")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples from TriviaQA")
    parser.add_argument("--split", type=str, default="train",
                        help="TriviaQA split")

    # Label mode
    parser.add_argument("--label_mode", type=str, default="binary",
                        choices=["binary", "uncertainty"],
                        help="Label mode: binary (temp=0) or uncertainty (temp>0, multi-trial)")
    parser.add_argument("--num_trials", type=int, default=10,
                        help="Number of trials for uncertainty mode")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for uncertainty mode")

    # Training mode
    parser.add_argument("--training_mode", type=str, default="batch",
                        choices=["batch", "adaptive"],
                        help="Training mode: batch or adaptive")

    # Training options
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")

    args = parser.parse_args()

    # Get world info
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank in [-1, 0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"\n{'='*60}")
        print("Judgment Training with DeepSpeed")
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"Output: {args.output_dir}")
        print(f"Label mode: {args.label_mode}")
        print(f"Training mode: {args.training_mode}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"World size: {world_size}")
        if args.label_mode == "uncertainty":
            print(f"Num trials: {args.num_trials}")
            print(f"Temperature: {args.temperature}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load or collect samples
    if args.input:
        if is_main:
            print(f"\nLoading samples from {args.input}...")
        samples = load_from_jsonl(args.input)
    else:
        if is_main:
            print(f"\nLoading TriviaQA {args.split} split...")
        samples = load_triviaqa(split=args.split, num_samples=args.num_samples)

        # Collect responses using all GPUs in parallel
        if is_main:
            print(f"\nCollecting responses ({args.label_mode} mode) with {world_size} GPUs...")

        # Each rank loads model on its own GPU
        if local_rank >= 0:
            device = f"cuda:{local_rank}"
        else:
            device = "cuda:0"

        inference_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)

        if args.label_mode == "binary":
            local_results = collect_responses_binary(
                samples, inference_model, tokenizer,
                batch_size=args.batch_size,
                local_rank=max(0, local_rank),
                world_size=world_size,
                show_progress=is_main
            )
        else:
            # TODO: Add distributed support for uncertainty mode
            local_results = collect_responses_uncertainty(
                samples, inference_model, tokenizer,
                num_trials=args.num_trials,
                batch_size=args.batch_size,
                temperature=args.temperature
            )

        # Free inference model memory
        del inference_model
        torch.cuda.empty_cache()

        # Save local results to per-rank file
        local_responses_file = output_dir / f"responses_rank{max(0, local_rank)}.jsonl"
        save_to_jsonl(local_results, str(local_responses_file))

        # Simple sync: wait for all ranks to finish
        if world_size > 1:
            import time
            # Create a done marker file
            done_file = output_dir / f".done_rank{max(0, local_rank)}"
            done_file.touch()

            # Wait for all ranks to be done
            for rank in range(world_size):
                other_done = output_dir / f".done_rank{rank}"
                while not other_done.exists():
                    time.sleep(0.5)

            # Only main process merges results
            if is_main:
                all_results = []
                for rank in range(world_size):
                    rank_file = output_dir / f"responses_rank{rank}.jsonl"
                    rank_results = load_from_jsonl(str(rank_file))
                    all_results.extend(rank_results)

                # Sort by original index to restore order
                all_results.sort(key=lambda x: x.get("_original_idx", 0))
                # Remove temporary field
                for r in all_results:
                    r.pop("_original_idx", None)

                samples = all_results

                # Save merged results
                responses_file = output_dir / "responses.jsonl"
                save_to_jsonl(samples, str(responses_file))
                print(f"Saved merged responses to {responses_file}")

                # Cleanup temp files
                for rank in range(world_size):
                    (output_dir / f"responses_rank{rank}.jsonl").unlink(missing_ok=True)
                    (output_dir / f".done_rank{rank}").unlink(missing_ok=True)

            # Other ranks load merged results
            else:
                time.sleep(1)  # Wait for main to merge
                responses_file = output_dir / "responses.jsonl"
                while not responses_file.exists():
                    time.sleep(0.5)
                time.sleep(1)  # Ensure file is fully written
                samples = load_from_jsonl(str(responses_file))
        else:
            samples = local_results

    # Print ability distribution
    if is_main:
        ability_counts = {"can": 0, "uncertain": 0, "cannot": 0}
        for s in samples:
            ability_counts[s["ability"]] = ability_counts.get(s["ability"], 0) + 1
        print(f"\nAbility distribution:")
        for ability, count in ability_counts.items():
            if count > 0:
                print(f"  {ability}: {count} ({count/len(samples)*100:.1f}%)")

    # Training loop
    all_stats = []
    current_model_path = args.model

    for epoch in range(1, args.epochs + 1):
        # Load model for training (no device_map for DeepSpeed compatibility)
        if is_main:
            print(f"\nLoading model from {current_model_path} for epoch {epoch}...")
        model = AutoModelForCausalLM.from_pretrained(
            current_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Run training
        if args.training_mode == "batch":
            tested_samples, avg_loss = run_batch_training(
                args, model, tokenizer, samples, epoch, is_main
            )
        else:
            tested_samples, avg_loss = run_adaptive_training(
                args, model, tokenizer, samples, epoch, is_main
            )

        # Save model
        epoch_dir = output_dir / f"epoch_{epoch}"
        if is_main:
            print(f"\nSaving model to {epoch_dir}...")

        # Save using trainer's save_model for DeepSpeed compatibility
        training_args = TrainingArguments(
            output_dir=str(epoch_dir),
            deepspeed=args.deepspeed,
        )
        trainer = Trainer(model=model, args=training_args)
        trainer.save_model(str(epoch_dir))
        tokenizer.save_pretrained(str(epoch_dir))

        # Update model path for next epoch and free memory
        current_model_path = str(epoch_dir)
        del model
        del trainer
        torch.cuda.empty_cache()

        # Save stats
        stats = {
            "epoch": epoch,
            "label_mode": args.label_mode,
            "training_mode": args.training_mode,
            "total_samples": len(samples),
            "avg_loss": avg_loss,
        }

        if tested_samples and args.training_mode == "batch":
            correct_count = sum(1 for s in tested_samples if s["judgment_correct"])
            stats["judgment_accuracy"] = correct_count / len(samples)
            stats["correct_count"] = correct_count
            stats["trained_count"] = len(samples) - correct_count

        all_stats.append(stats)

        if is_main:
            stats_file = epoch_dir / f"epoch_{epoch}_stats.json"
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"Epoch {epoch} complete! Loss: {avg_loss:.4f}")

    # Save final stats
    if is_main:
        final_stats = {
            "model": args.model,
            "label_mode": args.label_mode,
            "training_mode": args.training_mode,
            "epochs": args.epochs,
            "per_epoch_stats": all_stats,
        }
        with open(output_dir / "training_stats.json", "w") as f:
            json.dump(final_stats, f, indent=2)

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Final model saved to: {output_dir}/epoch_{args.epochs}")


if __name__ == "__main__":
    main()
