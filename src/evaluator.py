"""
Evaluator module - check if model responses are correct.
"""

import re
from typing import List, Dict
from collections import Counter


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


def classify_ability(accuracy: float, threshold_uncertain: float = 1.0, threshold_cannot: float = 0.5) -> str:
    """
    Classify model's ability based on accuracy.

    Args:
        accuracy: Accuracy across multiple trials
        threshold_uncertain: Below this = uncertain (default: 100% means any error = uncertain)
        threshold_cannot: Below this = cannot

    Returns:
        "can" / "uncertain" / "cannot"
    """
    if accuracy >= threshold_uncertain:
        return "can"
    elif accuracy >= threshold_cannot:
        return "uncertain"
    else:
        return "cannot"


def evaluate_samples(samples: List[Dict]) -> List[Dict]:
    """
    Evaluate all samples and add classification.

    Args:
        samples: List of samples with 'responses' and 'answers' fields

    Returns:
        Samples with added evaluation results
    """
    results = []
    for sample in samples:
        eval_result = evaluate_responses(
            sample["responses"],
            sample.get("normalized_answers", sample["answers"])
        )

        ability = classify_ability(eval_result["accuracy"])

        result = sample.copy()
        result["evaluation"] = eval_result
        result["ability"] = ability
        results.append(result)

    return results
