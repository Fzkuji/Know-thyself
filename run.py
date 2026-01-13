"""
Know-thyself: Main entry point for running the full pipeline.

Usage:
    python run.py --step 1  # Collect responses
    python run.py --step 2  # Build training dataset
    python run.py --step 3  # Train model
    python run.py --step all  # Run full pipeline
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(step: int, args: argparse.Namespace):
    """Run a specific step of the pipeline."""
    scripts_dir = Path(__file__).parent / "scripts"

    if step == 1:
        cmd = [
            sys.executable,
            str(scripts_dir / "step1_collect_responses.py"),
            "--model", args.model,
            "--num_samples", str(args.num_samples),
            "--num_trials", str(args.num_trials),
        ]
    elif step == 2:
        cmd = [
            sys.executable,
            str(scripts_dir / "step2_build_dataset.py"),
        ]
        if args.include_reason:
            cmd.append("--include_reason")
    elif step == 3:
        cmd = [
            sys.executable,
            str(scripts_dir / "step3_train.py"),
            "--model", args.model,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
        ]
    else:
        raise ValueError(f"Unknown step: {step}")

    print(f"\n{'='*60}")
    print(f"Running Step {step}")
    print(f"{'='*60}\n")

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Know-thyself: LLM Metacognition Training")
    parser.add_argument("--step", type=str, default="all", help="Step to run (1/2/3/all)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--include_reason", action="store_true")
    args = parser.parse_args()

    if args.step == "all":
        for step in [1, 2, 3]:
            run_step(step, args)
    else:
        run_step(int(args.step), args)

    print("\n" + "="*60)
    print("Pipeline completed!")
    print("="*60)


if __name__ == "__main__":
    main()
