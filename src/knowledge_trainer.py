"""
Knowledge Trainer - Build QA training data and train model to learn factual knowledge.

Unlike judgment training (which teaches metacognition), this module teaches
the model to actually answer questions correctly.
"""

from typing import List, Dict
from datasets import Dataset


# System prompt for QA training (simple and direct)
QA_SYSTEM_PROMPT = "You are a helpful assistant. Answer the question directly and concisely."


def build_qa_training_sample(question: str, answer: str) -> Dict:
    """
    Build a single QA training sample.

    Args:
        question: The question text
        answer: The correct answer

    Returns:
        Training sample in chat format
    """
    return {
        "messages": [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ],
        "task_type": "qa"
    }


def build_qa_dataset(samples: List[Dict]) -> List[Dict]:
    """
    Build QA training dataset from samples.

    Uses all samples regardless of ability classification.
    Each sample should have 'question' and either 'normalized_answers' or 'answers'.

    Args:
        samples: List of samples with question and answers

    Returns:
        QA training samples in chat format
    """
    qa_data = []

    for sample in samples:
        question = sample["question"]

        # Use first normalized answer as target
        if "normalized_answers" in sample and sample["normalized_answers"]:
            answer = sample["normalized_answers"][0]
        elif "answers" in sample and sample["answers"]:
            answer = sample["answers"][0]
        else:
            continue  # Skip if no answer available

        qa_sample = build_qa_training_sample(question=question, answer=answer)
        qa_sample["id"] = sample.get("id", "")

        # Preserve original ability if available
        if "ability" in sample:
            qa_sample["original_ability"] = sample["ability"]

        qa_data.append(qa_sample)

    return qa_data


def build_qa_dataset_filtered(
    samples: List[Dict],
    filter_abilities: List[str] = None
) -> List[Dict]:
    """
    Build QA training dataset with optional filtering by ability.

    Args:
        samples: List of samples with question, answers, and ability
        filter_abilities: List of abilities to include (e.g., ["cannot", "uncertain"])
                         If None, include all samples

    Returns:
        Filtered QA training samples
    """
    if filter_abilities is None:
        return build_qa_dataset(samples)

    filtered_samples = [
        s for s in samples
        if s.get("ability") in filter_abilities
    ]

    return build_qa_dataset(filtered_samples)
