"""
Label generator for HotpotQA - binary classification (can / cannot).
Output format uses <answer></answer> tags.
"""

from typing import List, Dict
from tqdm import tqdm

# System prompt for HotpotQA metacognition assessment (binary)
SYSTEM_PROMPT = """You are given a context and a question. First, assess whether you can answer the question correctly based on the context. Then provide your answer.

Output format:
1. First line: your judgment in \\boxed{}, using ONLY one word: yes (I can answer correctly) or no (I cannot answer correctly)
2. Second line: your answer wrapped in <answer></answer> tags

Example output:
\\boxed{yes}
<answer>Paris</answer>"""

# Label templates for binary classification
LABEL_TEMPLATES = {
    "can": "\\boxed{yes}\n<answer>{answer}</answer>",
    "cannot": "\\boxed{no}\n<answer>I don't know</answer>",
}


def generate_label(ability: str, answer: str = "") -> str:
    """
    Generate label text based on ability classification.

    Args:
        ability: "can" / "cannot"
        answer: The correct answer (used when ability is "can")

    Returns:
        Label text
    """
    if ability == "can":
        return LABEL_TEMPLATES["can"].format(answer=answer)
    else:
        return LABEL_TEMPLATES["cannot"]


def build_training_sample(context: str, question: str, ability: str, answer: str = "") -> Dict:
    """
    Build a single training sample for HotpotQA metacognition.

    Args:
        context: The context passages
        question: The question text
        ability: Model's ability classification ("can" / "cannot")
        answer: The correct answer

    Returns:
        Training sample with messages
    """
    label = generate_label(ability, answer)

    user_content = f"""Context:
{context}

Question: {question}

Can you answer this question correctly based on the context? Provide your judgment and answer."""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": label}
        ],
        "question": question,
        "context": context,
        "answer": answer,
        "ability": ability,
    }


def build_training_dataset(samples: List[Dict]) -> List[Dict]:
    """
    Build training dataset from evaluated samples.

    Args:
        samples: List of samples with 'question', 'context', 'answer', and 'ability' fields

    Returns:
        List of training samples
    """
    training_data = []
    for sample in tqdm(samples, desc="Building training data"):
        train_sample = build_training_sample(
            context=sample["context"],
            question=sample["question"],
            ability=sample["ability"],
            answer=sample.get("answer", ""),
        )
        train_sample["id"] = sample.get("id", "")
        training_data.append(train_sample)

    return training_data
