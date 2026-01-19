"""
Step 3 Binary: Train model for one epoch (binary judgment training).

Binary version:
- Only "can" and "cannot" (no "uncertain")
- Temperature=0 for all inference (greedy decoding)
- Single trial per question (correct or not)

Supports two modes:
- Batch mode (default): Pre-filter samples at epoch start, then batch train
- Sequential mode: Test each sample before training (slower but more adaptive)

This script trains the model for exactly ONE epoch, then exits.
Use this in a shell loop to interleave training and evaluation.
"""

import argparse
import os
import sys
import json
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm
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
from src.label_generator import SYSTEM_PROMPT
from src.evaluator import is_correct


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


def classify_ability_binary(correct: bool) -> str:
    """Binary classification: correct -> can, incorrect -> cannot."""
    return "can" if correct else "cannot"


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


def parse_judgment_response(response: str) -> str:
    """Parse judgment response to extract predicted ability."""
    response = response.strip().lower()

    # Parse \boxed{} format
    match = re.search(r'\\boxed\{(\w+)\}', response)
    if match:
        answer = match.group(1).lower()
        if answer == "yes":
            return "can"
        else:
            return "cannot"

    # Fallback: check keywords
    if "yes" in response:
        return "can"
    else:
        return "cannot"


def batch_generate_qa_responses(model, tokenizer, questions: list, batch_size: int = 16) -> list:
    """Generate QA responses in batches with greedy decoding."""
    all_responses = []

    for i in tqdm(range(0, len(questions), batch_size), desc="Generating QA responses"):
        batch_questions = questions[i:i + batch_size]

        # Build prompts
        prompts = []
        for q in batch_questions:
            messages = [
                {"role": "system", "content": "Answer the question directly and concisely."},
                {"role": "user", "content": q}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        # Generate
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode responses
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
            response = tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True
            ).strip()
            all_responses.append(response)

    return all_responses


def batch_generate_judgment_responses(model, tokenizer, questions: list, system_prompt: str, batch_size: int = 16) -> list:
    """Generate judgment responses in batches with greedy decoding."""
    all_responses = []

    for i in tqdm(range(0, len(questions), batch_size), desc="Generating judgment responses"):
        batch_questions = questions[i:i + batch_size]

        # Build prompts
        prompts = []
        for q in batch_questions:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Can you answer this question correctly?\n\nQuestion: {q}"}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        # Generate
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode responses
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
            response = tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True
            ).strip()
            all_responses.append(response)

    return all_responses


def preprocess_function(examples, tokenizer, system_prompt):
    """Preprocess samples for training."""
    texts = []
    for i in range(len(examples["question"])):
        question = examples["question"][i]
        ability = examples["ability"][i]

        sample = build_judgment_sample_binary(question, ability, system_prompt)
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


def filter_samples_batch(
    model,
    tokenizer,
    samples: list,
    system_prompt: str,
    inference_batch_size: int = 16,
    use_realtime_labels: bool = True,
    skip_correct: bool = True,
) -> tuple:
    """
    Filter samples using batch inference.

    Returns:
        tuple: (filtered_samples, stats)
            filtered_samples: list of samples that need training
            stats: dict with filtering statistics
    """
    stats = {
        "total": len(samples),
        "skipped": 0,
        "to_train": 0,
        "by_ability": {"can": 0, "cannot": 0},
    }

    questions = [s.get("question", "") for s in samples]
    gold_answers_list = [s.get("normalized_answers", s.get("answers", [])) for s in samples]

    # Step 1: Compute real-time abilities if needed
    if use_realtime_labels:
        print("Computing real-time abilities (batch QA inference)...")
        qa_responses = batch_generate_qa_responses(model, tokenizer, questions, inference_batch_size)

        abilities = []
        for response, gold_answers in zip(qa_responses, gold_answers_list):
            correct = is_correct(response, gold_answers)
            abilities.append(classify_ability_binary(correct))
    else:
        # Use pre-computed labels (convert to binary)
        abilities = []
        for s in samples:
            orig_ability = s.get("ability", "cannot")
            abilities.append("can" if orig_ability == "can" else "cannot")

    # Step 2: Get current judgment predictions if skip_correct
    if skip_correct:
        print("Getting current judgment predictions (batch inference)...")
        judgment_responses = batch_generate_judgment_responses(
            model, tokenizer, questions, system_prompt, inference_batch_size
        )
        predicted_abilities = [parse_judgment_response(r) for r in judgment_responses]
    else:
        predicted_abilities = [None] * len(samples)

    # Step 3: Filter samples
    filtered_samples = []
    for i, sample in enumerate(samples):
        ability = abilities[i]

        # Skip if already correct
        if skip_correct and predicted_abilities[i] == ability:
            stats["skipped"] += 1
            continue

        # Add to training set with computed ability
        filtered_sample = sample.copy()
        filtered_sample["ability"] = ability
        filtered_samples.append(filtered_sample)

        stats["to_train"] += 1
        stats["by_ability"][ability] += 1

    return filtered_samples, stats


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
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--inference_batch_size", type=int, default=16,
                        help="Batch size for inference (filtering)")
    parser.add_argument("--skip_correct", action="store_true", default=True,
                        help="Skip samples where model already judges correctly")
    parser.add_argument("--no_skip_correct", action="store_true",
                        help="Disable skip_correct (train on all samples)")
    parser.add_argument("--use_realtime_labels", action="store_true", default=True,
                        help="Compute ability labels in real-time (single greedy trial)")
    parser.add_argument("--no_realtime_labels", action="store_true",
                        help="Use pre-computed labels instead of real-time")
    args = parser.parse_args()

    # Handle negation flags
    skip_correct = args.skip_correct and not args.no_skip_correct
    use_realtime_labels = args.use_realtime_labels and not args.no_realtime_labels

    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"Epoch {args.epoch}: Binary Batch Training")
    print(f"{'='*60}")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Training data: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Learning rate: {args.lr}")
    print(f"Training batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Inference batch size: {args.inference_batch_size}")
    print(f"Mode: Binary (can/cannot), temperature=0")
    print(f"Skip correct: {skip_correct}")
    print(f"Real-time labels: {use_realtime_labels}")

    # Load training data
    print(f"\nLoading training data...")
    training_data = load_from_jsonl(args.input)
    print(f"Loaded {len(training_data)} samples")

    # Load model
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

    # Filter samples using batch inference
    print(f"\nFiltering samples (batch mode)...")
    filtered_samples, filter_stats = filter_samples_batch(
        model=model,
        tokenizer=tokenizer,
        samples=training_data,
        system_prompt=SYSTEM_PROMPT,
        inference_batch_size=args.inference_batch_size,
        use_realtime_labels=use_realtime_labels,
        skip_correct=skip_correct,
    )

    print(f"\nFiltering results:")
    print(f"  Total samples: {filter_stats['total']}")
    print(f"  Skipped (already correct): {filter_stats['skipped']}")
    print(f"  To train: {filter_stats['to_train']}")
    print(f"  By ability: can={filter_stats['by_ability']['can']}, cannot={filter_stats['by_ability']['cannot']}")

    # Check if there are samples to train
    if len(filtered_samples) == 0:
        print("\nNo samples to train! All samples are already correct.")
        print("Saving model without training...")
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Save stats
        stats = {
            "epoch": args.epoch,
            "total": filter_stats["total"],
            "skipped": filter_stats["skipped"],
            "trained": 0,
            "by_ability_trained": filter_stats["by_ability"],
        }
        stats_file = Path(args.output_dir) / f"epoch_{args.epoch}_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to {stats_file}")
        print(f"\nEpoch {args.epoch} complete (no training needed)!")
        return

    # Convert filtered samples to HF Dataset
    dataset_dict = {
        "question": [s.get("question", "") for s in filtered_samples],
        "ability": [s.get("ability", "cannot") for s in filtered_samples],
    }
    dataset = Dataset.from_dict(dataset_dict)

    # Preprocess dataset
    print(f"\nPreprocessing {len(filtered_samples)} samples for training...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, SYSTEM_PROMPT),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
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
        tokenizer=tokenizer,
    )

    # Train
    print(f"\nTraining on {len(filtered_samples)} samples...")
    train_result = trainer.train()

    # Save model
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save stats
    stats = {
        "epoch": args.epoch,
        "total": filter_stats["total"],
        "skipped": filter_stats["skipped"],
        "trained": filter_stats["to_train"],
        "by_ability_trained": filter_stats["by_ability"],
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
