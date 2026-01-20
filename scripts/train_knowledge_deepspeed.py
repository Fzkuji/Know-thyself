"""
Knowledge Training with DeepSpeed ZeRO-3.

Trains the model to answer questions correctly (factual knowledge).
This is separate from judgment training which teaches metacognition.

Usage:
    deepspeed --num_gpus=8 scripts/train_knowledge_deepspeed.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --input data/responses.jsonl \
        --output_dir experiments/knowledge \
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

from src.data_loader import load_triviaqa
from src.dataset_builder import load_from_jsonl, save_to_jsonl
from src.knowledge_trainer import build_qa_dataset, QA_SYSTEM_PROMPT


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


def preprocess_qa_function(examples, tokenizer):
    """Preprocess QA samples for training."""
    texts = []
    for i in range(len(examples["question"])):
        question = examples["question"][i]
        answer = examples["answer"][i]

        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]

        text = tokenizer.apply_chat_template(
            messages,
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

    # Filter options
    parser.add_argument("--filter_ability", type=str, nargs="+", default=None,
                        help="Filter samples by ability (e.g., --filter_ability cannot uncertain)")

    # Training options
    parser.add_argument("--epochs", type=int, default=1,
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
        print("Knowledge Training with DeepSpeed")
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"Output: {args.output_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"World size: {world_size}")
        if args.filter_ability:
            print(f"Filter abilities: {args.filter_ability}")

    # Load model and tokenizer
    if is_main:
        print(f"\nLoading model...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load samples
    if args.input:
        if is_main:
            print(f"\nLoading samples from {args.input}...")
        samples = load_from_jsonl(args.input)
    else:
        if is_main:
            print(f"\nLoading TriviaQA {args.split} split...")
        samples = load_triviaqa(split=args.split, num_samples=args.num_samples)

    if is_main:
        print(f"Loaded {len(samples)} samples")

    # Filter by ability if specified
    if args.filter_ability and "ability" in samples[0]:
        original_count = len(samples)
        samples = [s for s in samples if s.get("ability") in args.filter_ability]
        if is_main:
            print(f"Filtered to {len(samples)} samples (from {original_count})")

    # Build QA dataset
    if is_main:
        print(f"\nBuilding QA dataset...")

    qa_data = build_qa_dataset(samples)
    if is_main:
        print(f"Created {len(qa_data)} QA samples")

    # Convert to HF Dataset
    dataset_dict = {
        "question": [],
        "answer": [],
    }

    for sample in qa_data:
        # Extract from messages
        question = sample["messages"][1]["content"]
        answer = sample["messages"][2]["content"]
        dataset_dict["question"].append(question)
        dataset_dict["answer"].append(answer)

    dataset = Dataset.from_dict(dataset_dict)

    # Preprocess
    if is_main:
        print(f"Preprocessing dataset...")

    tokenized_dataset = dataset.map(
        lambda x: preprocess_qa_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Training
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
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
        print(f"\nTraining...")

    train_result = trainer.train()

    # Save model
    if is_main:
        print(f"\nSaving model to {output_dir}...")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save stats
    if is_main:
        stats = {
            "model": args.model,
            "total_samples": len(samples),
            "qa_samples": len(qa_data),
            "epochs": args.epochs,
            "filter_ability": args.filter_ability,
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        }
        with open(output_dir / "training_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n{'='*60}")
        print("Knowledge Training Complete!")
        print(f"{'='*60}")
        print(f"Loss: {train_result.training_loss:.4f}")
        print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
