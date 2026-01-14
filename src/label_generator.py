"""
Label generator - create training labels for metacognition.
"""

from typing import List, Dict, Optional

# System prompt for metacognition assessment
# Strict criteria: yes = 100% sure, no = 0% chance, uncertain = anything in between
SYSTEM_PROMPT = "Assess whether you can answer the question correctly. Output your judgment in \\boxed{}, using ONLY one word: yes (I am 100% certain I know the answer), uncertain (I might know but not sure), or no (I definitely don't know the answer)."

# Label templates (using \boxed{} format)
LABEL_TEMPLATES = {
    "can": "\\boxed{yes}",
    "uncertain": "\\boxed{uncertain}",
    "cannot": "\\boxed{no}",
}

# Extended templates with simple reasoning
LABEL_TEMPLATES_WITH_REASON = {
    "can": "I have sufficient knowledge about this topic. \\boxed{yes}",
    "uncertain": "I'm not fully confident about this. \\boxed{uncertain}",
    "cannot": "I lack the necessary knowledge. \\boxed{no}",
}


def generate_label(ability: str, include_reason: bool = False) -> str:
    """
    Generate label text based on ability classification.

    Args:
        ability: "can" / "uncertain" / "cannot"
        include_reason: Whether to include reasoning

    Returns:
        Label text
    """
    templates = LABEL_TEMPLATES_WITH_REASON if include_reason else LABEL_TEMPLATES
    return templates.get(ability, LABEL_TEMPLATES["cannot"])


def build_training_sample(question: str, ability: str, include_reason: bool = False) -> Dict:
    """
    Build a single training sample for metacognition.

    Args:
        question: The question text
        ability: Model's ability classification
        include_reason: Whether to include reasoning in label

    Returns:
        Training sample with input and output (using chat format)
    """
    label = generate_label(ability, include_reason)

    # Use chat format for training
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Can you answer this question correctly?\n\nQuestion: {question}"},
            {"role": "assistant", "content": label}
        ],
        "ability": ability,
    }


def build_training_dataset(samples: List[Dict], include_reason: bool = False) -> List[Dict]:
    """
    Build training dataset from evaluated samples.

    Args:
        samples: List of samples with 'question' and 'ability' fields
        include_reason: Whether to include reasoning

    Returns:
        List of training samples
    """
    training_data = []
    for sample in samples:
        train_sample = build_training_sample(
            question=sample["question"],
            ability=sample["ability"],
            include_reason=include_reason,
        )
        train_sample["id"] = sample.get("id", "")
        training_data.append(train_sample)

    return training_data
