"""
Data loader for TriviaQA dataset.
"""

from datasets import load_dataset
from typing import List, Dict, Optional


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

    samples = []
    for i, item in enumerate(dataset):
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


def format_question_prompt(question: str) -> str:
    """
    Format question into a prompt for the model.
    """
    return f"Question: {question}\nAnswer:"


if __name__ == "__main__":
    # Test loading
    samples = load_triviaqa("validation", num_samples=5)
    for s in samples:
        print(f"Q: {s['question']}")
        print(f"A: {s['answers'][:3]}")
        print("-" * 50)
