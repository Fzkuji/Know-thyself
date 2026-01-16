"""
Step 1: Collect model responses for each question (query 5 times).
Supports batch inference for better GPU utilization.
Supports multi-GPU inference with --multi_gpu flag.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_triviaqa, format_question_prompt
from src.multi_gpu_inference import create_inference
from src.evaluator import evaluate_samples
from src.dataset_builder import save_to_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of questions")
    parser.add_argument("--num_trials", type=int, default=5, help="Queries per question")
    parser.add_argument("--inference_batch_size", type=int, default=8, help="Batch size for inference")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--output", type=str, default=str(project_root / "data/step1_responses.jsonl"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multi-GPU inference")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (default: all)")
    args = parser.parse_args()

    print(f"Loading TriviaQA {args.split} split...")
    samples = load_triviaqa(split=args.split, num_samples=args.num_samples)
    print(f"Loaded {len(samples)} samples")

    print(f"Loading model: {args.model}")
    print(f"Inference batch size: {args.inference_batch_size}")
    print(f"Multi-GPU: {args.multi_gpu}")
    model = create_inference(
        model_name=args.model,
        inference_batch_size=args.inference_batch_size,
        multi_gpu=args.multi_gpu,
        num_gpus=args.num_gpus,
    )

    print(f"Running inference ({args.num_trials} trials per question)...")
    results = model.batch_inference(
        samples,
        num_trials=args.num_trials,
        prompt_formatter=lambda s: format_question_prompt(s["question"]),
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

    # Cleanup multi-GPU resources
    if hasattr(model, 'shutdown'):
        model.shutdown()

    print("Done!")


if __name__ == "__main__":
    main()
