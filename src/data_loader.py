"""
Data loader for QA datasets (TriviaQA, HotpotQA).
"""

from datasets import load_dataset
from typing import List, Dict, Optional
from tqdm import tqdm


SUPPORTED_DATASETS = ["triviaqa", "hotpotqa"]


def load_dataset_by_name(
    dataset_name: str,
    split: str = "train",
    num_samples: Optional[int] = None
) -> List[Dict]:
    """
    Load a QA dataset by name.

    Args:
        dataset_name: Name of the dataset ("triviaqa" or "hotpotqa")
        split: Dataset split ("train", "validation", "test")
        num_samples: Number of samples to load (None for all)

    Returns:
        List of samples with unified schema:
        - id: Unique identifier
        - question: The question text
        - answers: List of acceptable answers (for evaluation)
        - normalized_answers: Normalized versions of answers
        - primary_answer: Primary/canonical answer
        - normalized_primary: Normalized primary answer
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "triviaqa":
        return load_triviaqa(split=split, num_samples=num_samples)
    elif dataset_name == "hotpotqa":
        return load_hotpotqa(split=split, num_samples=num_samples)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {SUPPORTED_DATASETS}")


def load_triviaqa(split: str = "train", num_samples: Optional[int] = None) -> List[Dict]:
    """
    Load TriviaQA dataset.

    Args:
        split: Dataset split ("train", "validation", "test")
        num_samples: Number of samples to load (None for all)

    Returns:
        List of samples with 'question' and 'answers' fields
    """
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)

    total = num_samples if num_samples else len(dataset)
    samples = []
    for i, item in tqdm(enumerate(dataset), total=total, desc=f"Loading {split} data"):
        if num_samples and i >= num_samples:
            break

        sample = {
            "id": item.get("question_id", str(i)),
            "question": item["question"],
            "answers": item["answer"]["aliases"],  # List of acceptable answers (for evaluation)
            "normalized_answers": item["answer"]["normalized_aliases"],  # For evaluation
            "primary_answer": item["answer"]["value"],  # Primary answer (for training)
            "normalized_primary": item["answer"]["normalized_value"],  # Normalized primary
        }
        samples.append(sample)

    return samples


def load_hotpotqa(split: str = "train", num_samples: Optional[int] = None) -> List[Dict]:
    """
    Load HotpotQA dataset.

    Args:
        split: Dataset split ("train", "validation")
        num_samples: Number of samples to load (None for all)

    Returns:
        List of samples with unified schema matching TriviaQA format
    """
    # Use distractor configuration (has train and validation splits)
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)

    total = num_samples if num_samples else len(dataset)
    samples = []
    for i, item in tqdm(enumerate(dataset), total=total, desc=f"Loading HotpotQA {split} data"):
        if num_samples and i >= num_samples:
            break

        answer = item["answer"]
        # Normalize answer (lowercase, strip whitespace)
        normalized_answer = answer.lower().strip()

        sample = {
            "id": item["id"],
            "question": item["question"],
            # HotpotQA has single answer, wrap in list for compatibility
            "answers": [answer],
            "normalized_answers": [normalized_answer],
            "primary_answer": answer,
            "normalized_primary": normalized_answer,
            # Additional HotpotQA-specific fields (optional)
            "type": item.get("type", ""),  # "comparison" or "bridge"
            "level": item.get("level", ""),  # "easy", "medium", "hard"
        }
        samples.append(sample)

    return samples


def format_question_prompt(question: str) -> str:
    """
    Format question into a prompt for the model.
    """
    return f"Question: {question}\nAnswer:"


if __name__ == "__main__":
    # Test TriviaQA loading
    print("=" * 60)
    print("Testing TriviaQA")
    print("=" * 60)
    samples = load_triviaqa("validation", num_samples=5)
    for s in samples:
        print(f"Q: {s['question']}")
        print(f"A: {s['answers'][:3]}")
        print("-" * 50)

    # Test HotpotQA loading
    print("\n" + "=" * 60)
    print("Testing HotpotQA")
    print("=" * 60)
    samples = load_hotpotqa("validation", num_samples=5)
    for s in samples:
        print(f"Q: {s['question']}")
        print(f"A: {s['answers']}")
        print(f"Type: {s['type']}, Level: {s['level']}")
        print("-" * 50)

    # Test unified loader
    print("\n" + "=" * 60)
    print("Testing unified loader")
    print("=" * 60)
    for dataset_name in SUPPORTED_DATASETS:
        samples = load_dataset_by_name(dataset_name, "validation", num_samples=2)
        print(f"\n{dataset_name.upper()}:")
        for s in samples:
            print(f"  Q: {s['question'][:60]}...")
            print(f"  A: {s['primary_answer']}")
