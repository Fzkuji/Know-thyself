"""
DeepSpeed ZeRO-3 training for judgment task.

Expects pre-tested samples with 'judgment_correct' field.
Only trains on samples where judgment was incorrect.

Usage:
    deepspeed --num_gpus=8 scripts/train_deepspeed.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --input experiments/judgment/tested.jsonl \
        --output_dir experiments/judgment/epoch_1 \
        --deepspeed configs/ds_config_zero3.json
"""

import argparse
import os
import sys
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

from src.dataset_builder import load_from_jsonl, save_to_jsonl
from src.label_generator import build_training_sample


@dataclass
class DataCollatorForCausalLM:
    """Data collator for causal language modeling.

    Properly handles left-padding by padding labels on the left as well,
    to maintain alignment between input_ids and labels.
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

            # Match padding side with tokenizer
            pad_left = getattr(self.tokenizer, "padding_side", "right") == "left"

            padded_labels = []
            for label in labels:
                remainder = max_label_length - len(label)
                if pad_left:
                    # Left padding: add -100 at the beginning
                    padded_label = [self.label_pad_token_id] * remainder + list(label)
                else:
                    # Right padding: add -100 at the end
                    padded_label = list(label) + [self.label_pad_token_id] * remainder
                padded_labels.append(padded_label)

            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


def preprocess_function(examples, tokenizer, label_mode="binary"):
    """Preprocess judgment samples for training.

    Only compute loss on assistant response (the judgment label),
    not on the system prompt or user question.
    """
    all_input_ids = []
    all_labels = []

    for i in range(len(examples["question"])):
        question = examples["question"][i]
        ability = examples["ability"][i]

        sample = build_training_sample(question, ability, label_mode=label_mode)

        # Tokenize the full conversation
        full_text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize without assistant response to find where it starts
        messages_without_assistant = sample["messages"][:-1]  # system + user only
        prompt_text = tokenizer.apply_chat_template(
            messages_without_assistant,
            tokenize=False,
            add_generation_prompt=True  # This adds the assistant turn start
        )

        # Tokenize both
        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=512,
            padding=False,
        )
        prompt_tokens = tokenizer(
            prompt_text,
            truncation=True,
            max_length=512,
            padding=False,
        )

        input_ids = full_tokens["input_ids"]
        prompt_len = len(prompt_tokens["input_ids"])

        # Create labels: -100 for prompt, actual tokens for response
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": [[1] * len(ids) for ids in all_input_ids],
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL with tested samples")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for trained model")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file")

    # Training options
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)

    # Filter options
    parser.add_argument("--only_incorrect", action="store_true",
                        help="Only train on incorrect judgments (default: train on all samples)")

    # Label mode
    parser.add_argument("--label_mode", type=str, default="binary", choices=["binary", "uncertainty"],
                        help="Label mode: binary (yes/no) or uncertainty (yes/uncertain/no)")

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    is_main = local_rank in [-1, 0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load samples
    samples = load_from_jsonl(args.input)

    # Filter samples
    if args.only_incorrect:
        # Only train on incorrect judgments
        train_samples = [s for s in samples if not s.get("judgment_correct", True)]
    else:
        # Train on all samples (to prevent forgetting)
        train_samples = samples

    if len(train_samples) == 0:
        if is_main:
            print("No samples to train on!")
        return

    # Build dataset
    dataset_dict = {
        "question": [s["question"] for s in train_samples],
        "ability": [s["ability"] for s in train_samples],
    }
    dataset = Dataset.from_dict(dataset_dict)

    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, label_mode=args.label_mode),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Load model (no device_map for DeepSpeed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
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

    # Save model (all ranks must participate for ZeRO-3)
    # For ZeRO-3, all ranks must call save_model to gather weights
    trainer.save_model(str(output_dir))

    if is_main:
        tokenizer.save_pretrained(str(output_dir))

    # Save stats
    if is_main:
        stats = {
            "input": args.input,
            "total_samples": len(samples),
            "trained_samples": len(train_samples),
            "training_loss": train_result.training_loss,
        }
        with open(output_dir / "training_stats.json", "w") as f:
            import json
            json.dump(stats, f, indent=2)

        print(f"\nTraining complete! Loss: {train_result.training_loss:.4f}")


if __name__ == "__main__":
    main()
