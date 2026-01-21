"""
Label generator - create training labels for metacognition.
"""

from typing import List, Dict, Optional
from tqdm import tqdm

# =============================================================================
# Binary mode: yes/no only
# =============================================================================
SYSTEM_PROMPT_BINARY = "Assess whether you can answer the question correctly. Output ONLY \\boxed{yes} or \\boxed{no}."

LABEL_TEMPLATES_BINARY = {
    "can": "\\boxed{yes}",
    "cannot": "\\boxed{no}",
}

# =============================================================================
# Uncertainty mode: yes/uncertain/no
# =============================================================================
SYSTEM_PROMPT_UNCERTAINTY = "Assess whether you can answer the question correctly. Output your judgment in \\boxed{}, using ONLY one word: yes (I am 100% certain I know the answer), uncertain (I might know but not sure), or no (I definitely don't know the answer)."

LABEL_TEMPLATES_UNCERTAINTY = {
    "can": "\\boxed{yes}",
    "uncertain": "\\boxed{uncertain}",
    "cannot": "\\boxed{no}",
}

# =============================================================================
# Default (for backward compatibility)
# =============================================================================
SYSTEM_PROMPT = SYSTEM_PROMPT_BINARY
LABEL_TEMPLATES = LABEL_TEMPLATES_BINARY

# Extended templates with simple reasoning (uncertainty mode only)
LABEL_TEMPLATES_WITH_REASON = {
    "can": "I have sufficient knowledge about this topic. \\boxed{yes}",
    "uncertain": "I'm not fully confident about this. \\boxed{uncertain}",
    "cannot": "I lack the necessary knowledge. \\boxed{no}",
}


def get_system_prompt(label_mode: str = "binary") -> str:
    """Get system prompt based on label mode."""
    if label_mode == "binary":
        return SYSTEM_PROMPT_BINARY
    else:
        return SYSTEM_PROMPT_UNCERTAINTY


def get_label_templates(label_mode: str = "binary") -> dict:
    """Get label templates based on label mode."""
    if label_mode == "binary":
        return LABEL_TEMPLATES_BINARY
    else:
        return LABEL_TEMPLATES_UNCERTAINTY


def generate_label(ability: str, label_mode: str = "binary", include_reason: bool = False) -> str:
    """
    Generate label text based on ability classification.

    Args:
        ability: "can" / "uncertain" / "cannot"
        label_mode: "binary" or "uncertainty"
        include_reason: Whether to include reasoning

    Returns:
        Label text
    """
    if include_reason:
        templates = LABEL_TEMPLATES_WITH_REASON
    else:
        templates = get_label_templates(label_mode)
    return templates.get(ability, templates.get("cannot", "\\boxed{no}"))


def build_training_sample(question: str, ability: str, label_mode: str = "binary", include_reason: bool = False) -> Dict:
    """
    Build a single training sample for metacognition.

    Args:
        question: The question text
        ability: Model's ability classification
        label_mode: "binary" or "uncertainty"
        include_reason: Whether to include reasoning in label

    Returns:
        Training sample with input and output (using chat format)
    """
    label = generate_label(ability, label_mode, include_reason)
    system_prompt = get_system_prompt(label_mode)

    # Use chat format for training
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Can you answer this question correctly?\n\nQuestion: {question}"},
            {"role": "assistant", "content": label}
        ],
        "question": question,  # Keep question for adaptive training
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
    for sample in tqdm(samples, desc="Building training data"):
        train_sample = build_training_sample(
            question=sample["question"],
            ability=sample["ability"],
            include_reason=include_reason,
        )
        train_sample["id"] = sample.get("id", "")
        training_data.append(train_sample)

    return training_data
