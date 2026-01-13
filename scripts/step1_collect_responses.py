"""
Step 1: Collect model responses for each question (query 5 times).
"""

import argparse
import sys
sys.path.append("..")

from src.data_loader import load_triviaqa, format_question_prompt
from src.inference import ModelInference
from src.evaluator import evaluate_samples
from src.dataset_builder import save_to_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of questions")
    parser.add_argument("--num_trials", type=int, default=5, help="Queries per question")
    parser.add_argument("--output", type=str, default="../data/step1_responses.jsonl")
    parser.add_argument("--split", type=str, default="validation")
    args = parser.parse_args()

    print(f"Loading TriviaQA {args.split} split...")
    samples = load_triviaqa(split=args.split, num_samples=args.num_samples)
    print(f"Loaded {len(samples)} samples")

    print(f"Loading model: {args.model}")
    model = ModelInference(model_name=args.model)

    print(f"Running inference ({args.num_trials} trials per question)...")
    results = model.batch_inference(
        samples,
        num_trials=args.num_trials,
        prompt_formatter=format_question_prompt,
    )

    print("Evaluating responses...")
    results = evaluate_samples(results)

    # Statistics
    ability_counts = {"can": 0, "uncertain": 0, "cannot": 0}
    for r in results:
        ability_counts[r["ability"]] += 1

    print(f"\nAbility distribution:")
    for ability, count in ability_counts.items():
        print(f"  {ability}: {count} ({count/len(results)*100:.1f}%)")

    print(f"\nSaving to {args.output}")
    save_to_jsonl(results, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
