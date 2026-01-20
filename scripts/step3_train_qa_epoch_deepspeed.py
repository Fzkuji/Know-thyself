"""
Step 3 QA DeepSpeed: Train model for QA (knowledge) for one epoch with DeepSpeed ZeRO-3.

This script trains the model to answer questions correctly (knowledge learning),
not judgment training. It's used in Phase 1.1 for alternating training.

Usage:
    deepspeed --num_gpus=8 scripts/step3_train_qa_epoch_deepspeed.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --input training_data.jsonl \
        --output_dir epoch_1_qa \
        --deepspeed configs/ds_config_zero3.json
"""

import argparse
import os
import sys
import json
import re
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

from src.dataset_builder import load_from_jsonl
from src.knowledge_trainer import build_qa_dataset


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
        # Separate labels from features
        labels = [feature.pop("labels") for feature in features] if "labels" in features[0] else None

        # Pad input_ids and attention_mask
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad labels separately with label_pad_token_id
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
                # Pad on the right with label_pad_token_id
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

        # Build QA messages
        messages = [
            {"role": "user", "content": f"Question: {question}\nAnswer:"},
            {"role": "assistant", "content": answer}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding=False,
    )

    # Labels = input_ids for causal LM
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model path (base model or continued from previous epoch)")
    parser.add_argument("--input", type=str, required=True,
                        help="Training data JSONL file with questions and answers")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory to save the model")
    parser.add_argument("--epoch", type=int, default=1,
                        help="Current epoch number (for logging)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    args = parser.parse_args()

    # Get world info
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank in [-1, 0]

    if is_main:
        print(f"\n{'='*60}")
        print(f"{'='*60}")
        print(f"Epoch {args.epoch}: QA Training (DeepSpeed ZeRO-3)")
        print(f"{'='*60}")
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"Training data: {args.input}")
        print(f"Output: {args.output_dir}")
        print(f"Learning rate: {args.lr}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"World size: {world_size}")
        print(f"DeepSpeed config: {args.deepspeed}")

    # Load training data
    if is_main:
        print(f"\nLoading training data...")
    training_data = load_from_jsonl(args.input)

    # Build QA dataset
    if is_main:
        print(f"Building QA dataset from {len(training_data)} samples...")
    qa_data = build_qa_dataset(training_data)
    if is_main:
        print(f"Created {len(qa_data)} QA training samples")

    # Convert to HF Dataset
    dataset_dict = {
        "question": [],
        "answer": [],
    }

    for sample in qa_data:
        # Extract question and answer from messages
        user_msg = sample["messages"][1]["content"]  # "Question: ... Answer:"
        assistant_msg = sample["messages"][2]["content"]  # The answer

        # Extract just the question part
        question = user_msg.replace("Question: ", "").replace("\nAnswer:", "").strip()

        dataset_dict["question"].append(question)
        dataset_dict["answer"].append(assistant_msg)

    dataset = Dataset.from_dict(dataset_dict)

    # Load tokenizer
    if is_main:
        print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Preprocess dataset
    if is_main:
        print(f"Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_qa_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing QA samples",
    )

    # Load model
    if is_main:
        print(f"\nLoading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # Don't use device_map with DeepSpeed
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
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

    # Data collator (custom one that properly pads labels)
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Train
    if is_main:
        print(f"\nTraining QA epoch {args.epoch}...")
    train_result = trainer.train()

    # Save model (DeepSpeed handles gathering weights)
    if is_main:
        print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save stats
    if is_main:
        stats = {
            "epoch": args.epoch,
            "type": "qa_training",
            "total_samples": len(training_data),
            "qa_samples": len(qa_data),
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        }
        stats_file = Path(args.output_dir) / f"epoch_{args.epoch}_qa_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to {stats_file}")
        print(f"\nQA Epoch {args.epoch} complete!")


if __name__ == "__main__":
    main()
