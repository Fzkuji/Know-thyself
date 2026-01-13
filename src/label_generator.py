"""
Label generator - create training labels for metacognition.
"""

from typing import List, Dict, Optional

# System prompt for metacognition assessment
SYSTEM_PROMPT = "You are assessing whether you can answer a question correctly. Reply with ONLY one of: 'I can answer this question' or 'I cannot answer this question'. Do not provide the actual answer."

# Label templates
LABEL_TEMPLATES = {
    "can": "I can answer this question.",
    "uncertain": "I cannot answer this question.",  # Map uncertain to cannot for simplicity
    "cannot": "I cannot answer this question.",
}

# Extended templates with simple reasoning (no external model analysis)
LABEL_TEMPLATES_WITH_REASON = {
    "can": "I can answer this question. I have sufficient knowledge to provide the correct answer.",
    "uncertain": "I cannot answer this question. I might know the answer but I'm not fully confident.",
    "cannot": "I cannot answer this question. I lack the necessary knowledge.",
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
