"""
Step 3 DeepSpeed: Train model for one epoch with DeepSpeed ZeRO-3.

Uses Hugging Face Trainer with DeepSpeed for proper multi-GPU training.
This solves the issue where device_map="auto" causes training problems.

Usage:
    deepspeed --num_gpus=8 scripts/step3_train_epoch_deepspeed.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --input training_data.jsonl \
        --output_dir epoch_1 \
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
    DataCollatorForLanguageModeling,
)

from src.dataset_builder import load_from_jsonl
from src.label_generator import SYSTEM_PROMPT
from src.evaluator import is_correct, classify_ability


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


def preprocess_function(examples, tokenizer, system_prompt):
    """Preprocess samples for training."""
    texts = []
    for i in range(len(examples["question"])):
        question = examples["question"][i]
        ability = examples["ability"][i]

        sample = build_judgment_sample(question, ability, system_prompt)
        text = tokenizer.apply_chat_template(
            sample["messages"],
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
                        help="Training data JSONL file")
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
    parser.add_argument("--num_trials", type=int, default=10,
                        help="Number of trials for ability classification (used for pre-computed labels)")
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
        print(f"Epoch {args.epoch}: DeepSpeed ZeRO-3 Training")
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
    if is_main:
        print(f"Loaded {len(training_data)} samples")

    # Convert to HF Dataset
    dataset_dict = {
        "question": [s.get("question", "") for s in training_data],
        "ability": [s.get("ability", "cannot") for s in training_data],
        "normalized_answers": [s.get("normalized_answers", s.get("answers", [])) for s in training_data],
    }
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
        lambda x: preprocess_function(x, tokenizer, SYSTEM_PROMPT),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
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

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    if is_main:
        print(f"\nTraining epoch {args.epoch}...")
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
            "total": len(training_data),
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        }
        stats_file = Path(args.output_dir) / f"epoch_{args.epoch}_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to {stats_file}")
        print(f"\nEpoch {args.epoch} complete!")


if __name__ == "__main__":
    main()
