"""
Evaluator module - check if model responses are correct.
"""

import re
from typing import List, Dict
from collections import Counter
from tqdm import tqdm


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = " ".join(text.split())
    return text


def is_correct(response: str, gold_answers: List[str]) -> bool:
    """
    Check if response matches any of the gold answers.

    Args:
        response: Model's response
        gold_answers: List of acceptable answers

    Returns:
        True if response matches any gold answer
    """
    norm_response = normalize_answer(response)

    for answer in gold_answers:
        norm_answer = normalize_answer(answer)
        # Exact match or containment
        if norm_answer in norm_response or norm_response in norm_answer:
            return True

    return False


def evaluate_responses(responses: List[str], gold_answers: List[str]) -> Dict:
    """
    Evaluate multiple responses against gold answers.

    Args:
        responses: List of model responses
        gold_answers: List of acceptable answers

    Returns:
        Dict with evaluation results
    """
    results = [is_correct(r, gold_answers) for r in responses]
    correct_count = sum(results)
    total = len(responses)
    accuracy = correct_count / total if total > 0 else 0.0

    return {
        "correct_count": correct_count,
        "total": total,
        "accuracy": accuracy,
        "per_response": results,
    }


def classify_ability(correct_count: int, total: int = 5) -> str:
    """
    Classify model's ability based on correct count.

    Strict criteria for 5 trials:
    - 5/5 correct = "can" (definitely knows)
    - 0/5 correct = "cannot" (definitely doesn't know)
    - 1-4/5 correct = "uncertain" (inconsistent)

    Args:
        correct_count: Number of correct responses
        total: Total number of trials (default: 5)

    Returns:
        "can" / "uncertain" / "cannot"
    """
    if correct_count == total:
        return "can"
    elif correct_count == 0:
        return "cannot"
    else:
        return "uncertain"


def evaluate_samples(samples: List[Dict]) -> List[Dict]:
    """
    Evaluate all samples and add classification.

    Args:
        samples: List of samples with 'responses' and 'answers' fields

    Returns:
        Samples with added evaluation results
    """
    results = []
    for sample in tqdm(samples, desc="Evaluating responses"):
        eval_result = evaluate_responses(
            sample["responses"],
            sample.get("normalized_answers", sample["answers"])
        )

        ability = classify_ability(eval_result["correct_count"], eval_result["total"])

        result = sample.copy()
        result["evaluation"] = eval_result
        result["ability"] = ability
        results.append(result)

    return results
