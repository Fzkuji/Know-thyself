"""
Step 2: Build training dataset from collected responses.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_builder import load_from_jsonl, save_to_jsonl
from src.label_generator import build_training_dataset


def main():
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--input", type=str, default=str(project_root / "data/step1_responses.jsonl"))
    parser.add_argument("--output", type=str, default=str(project_root / "data/step2_training_data.jsonl"))
    parser.add_argument("--include_reason", action="store_true", help="Include reasoning in labels")
    args = parser.parse_args()

    print(f"Loading responses from {args.input}")
    samples = load_from_jsonl(args.input)
    print(f"Loaded {len(samples)} samples")

    print("Building training dataset...")
    training_data = build_training_dataset(samples, include_reason=args.include_reason)

    # Show examples
    print("\n--- Example training samples ---")
    for i, sample in enumerate(training_data[:3]):
        print(f"\n[{i+1}] Ability: {sample['ability']}")
        print(f"Input: {sample['input'][:100]}...")
        print(f"Output: {sample['output']}")

    print(f"\nSaving to {args.output}")
    save_to_jsonl(training_data, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
