"""
Dataset builder - create HuggingFace datasets for training.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datasets import Dataset


def save_to_jsonl(data: List[Dict], path: str):
    """Save data to JSONL format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_from_jsonl(path: str) -> List[Dict]:
    """Load data from JSONL format."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_hf_dataset(training_data: List[Dict]) -> Dataset:
    """
    Convert training data to HuggingFace Dataset.

    Args:
        training_data: List of training samples

    Returns:
        HuggingFace Dataset
    """
    return Dataset.from_list(training_data)


def format_for_sft(sample: Dict, tokenizer=None) -> Dict:
    """
    Format sample for supervised fine-tuning.

    Args:
        sample: Training sample with 'messages' or 'input'/'output'
        tokenizer: Optional tokenizer for chat template

    Returns:
        Formatted sample
    """
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        # Use messages if available, otherwise create from input/output
        if "messages" in sample:
            messages = sample["messages"]
        else:
            messages = [
                {"role": "user", "content": sample["input"]},
                {"role": "assistant", "content": sample["output"]},
            ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Fallback for non-chat models
        if "messages" in sample:
            # Extract from messages
            user_msg = next((m["content"] for m in sample["messages"] if m["role"] == "user"), "")
            assistant_msg = next((m["content"] for m in sample["messages"] if m["role"] == "assistant"), "")
            text = f"### Instruction:\n{user_msg}\n\n### Response:\n{assistant_msg}"
        else:
            text = f"### Instruction:\n{sample['input']}\n\n### Response:\n{sample['output']}"

    return {"text": text}


def prepare_dataset_for_training(
    training_data: List[Dict],
    tokenizer=None,
    test_size: float = 0.1,
) -> Dict:
    """
    Prepare dataset splits for training.

    Args:
        training_data: List of training samples
        tokenizer: Tokenizer for formatting
        test_size: Fraction for validation

    Returns:
        Dict with 'train' and 'validation' datasets
    """
    dataset = create_hf_dataset(training_data)

    # Format for SFT
    dataset = dataset.map(
        lambda x: format_for_sft(x, tokenizer),
        desc="Formatting for SFT"
    )

    # Split
    splits = dataset.train_test_split(test_size=test_size, seed=42)

    return {
        "train": splits["train"],
        "validation": splits["test"],
    }
