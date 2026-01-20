"""
Data loader for HotpotQA dataset (distractor config).
"""

from datasets import load_dataset
from typing import List, Dict, Optional
from tqdm import tqdm


def load_hotpotqa(split: str = "train", num_samples: Optional[int] = None) -> List[Dict]:
    """
    Load HotpotQA dataset (distractor config).

    Args:
        split: Dataset split ("train", "validation")
        num_samples: Number of samples to load (None for all)

    Returns:
        List of samples with 'question', 'answer', and 'context' fields
    """
    dataset = load_dataset("hotpot_qa", "distractor", split=split)

    total = num_samples if num_samples else len(dataset)
    samples = []
    for i, item in tqdm(enumerate(dataset), total=total, desc=f"Loading {split} data"):
        if num_samples and i >= num_samples:
            break

        # Build context from titles and sentences
        context_parts = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            # Join sentences for each article
            article_text = " ".join(sentences)
            context_parts.append(f"[{title}] {article_text}")

        context = "\n\n".join(context_parts)

        sample = {
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "answers": [item["answer"]],  # For compatibility with evaluator
            "normalized_answers": [item["answer"].lower()],
            "context": context,
            "type": item["type"],
            "level": item["level"],
        }
        samples.append(sample)

    return samples


def format_hotpotqa_prompt(sample: Dict) -> str:
    """
    Format HotpotQA sample into a prompt for the model.
    Output should be wrapped in <answer>...</answer> tags.
    """
    context = sample["context"]
    question = sample["question"]

    prompt = f"""Based on the following context, answer the question. Wrap your answer in <answer></answer> tags.

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def extract_answer_from_response(response: str) -> str:
    """
    Extract answer from <answer>...</answer> tags.
    """
    import re
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return the whole response
    return response.strip()


if __name__ == "__main__":
    # Test loading
    samples = load_hotpotqa("validation", num_samples=3)
    for s in samples:
        print(f"Q: {s['question']}")
        print(f"A: {s['answer']}")
        print(f"Context (first 200 chars): {s['context'][:200]}...")
        print("-" * 50)
