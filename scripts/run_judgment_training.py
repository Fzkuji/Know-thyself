#!/usr/bin/env python3
"""
Main script to run judgment training pipeline.

Orchestrates:
1. Collect responses (data parallel inference)
2. Evaluate baseline QA accuracy (before training)
3. For each epoch:
   a. Test judgments (data parallel inference)
   b. Train on samples (DeepSpeed ZeRO-3)
   c. Evaluate QA accuracy (monitor for degradation)

Usage:
    python scripts/run_judgment_training.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output_dir experiments/judgment \
        --epochs 10 \
        --num_gpus 8
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, desc):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: {desc} failed with code {result.returncode}")
        sys.exit(1)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--label_mode", type=str, default="binary", choices=["binary", "uncertainty"])
    parser.add_argument("--deepspeed_config", type=str, default="configs/ds_config_zero3.json")

    # Resume options
    parser.add_argument("--skip_collect", action="store_true", help="Skip response collection (use existing)")
    parser.add_argument("--start_epoch", type=int, default=1, help="Start from this epoch")
    parser.add_argument("--skip_qa_eval", action="store_true", help="Skip QA accuracy evaluation")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).parent
    responses_file = output_dir / "responses.jsonl"

    print(f"\n{'#'*60}")
    print("Judgment Training Pipeline")
    print(f"{'#'*60}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Label mode: {args.label_mode}")

    # Step 1: Collect responses (only if not skipped)
    if not args.skip_collect and not responses_file.exists():
        cmd = [
            "torchrun", f"--nproc_per_node={args.num_gpus}",
            str(scripts_dir / "inference_ddp.py"),
            "--mode", "collect",
            "--model", args.model,
            "--output_dir", str(output_dir),
            "--num_samples", str(args.num_samples),
            "--batch_size", str(args.batch_size),
            "--label_mode", args.label_mode,
        ]
        run_command(cmd, "Collect responses")
    else:
        print(f"\nSkipping response collection (using {responses_file})")

    # Step 2: Evaluate baseline QA accuracy (before training)
    if not args.skip_qa_eval and args.start_epoch == 1:
        cmd = [
            "torchrun", f"--nproc_per_node={args.num_gpus}",
            str(scripts_dir / "inference_ddp.py"),
            "--mode", "eval_qa",
            "--model", args.model,
            "--input", str(responses_file),
            "--output_dir", str(output_dir),
            "--batch_size", str(args.batch_size),
            "--output_file", "eval_qa_epoch0.jsonl",
        ]
        run_command(cmd, "Evaluate baseline QA accuracy (epoch 0)")

    # Training loop
    current_model = args.model

    for epoch in range(args.start_epoch, args.epochs + 1):
        print(f"\n{'#'*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'#'*60}")

        tested_file = output_dir / f"tested_epoch{epoch}.jsonl"
        epoch_dir = output_dir / f"epoch_{epoch}"

        # Step 3a: Test judgments
        cmd = [
            "torchrun", f"--nproc_per_node={args.num_gpus}",
            str(scripts_dir / "inference_ddp.py"),
            "--mode", "test",
            "--model", current_model,
            "--input", str(responses_file),
            "--output_dir", str(output_dir),
            "--batch_size", str(args.batch_size),
            "--output_file", f"tested_epoch{epoch}.jsonl",
            "--label_mode", args.label_mode,
        ]
        run_command(cmd, f"Test judgments (epoch {epoch})")

        # Step 3b: Train on all samples (prevents forgetting correct judgments)
        cmd = [
            "deepspeed", f"--num_gpus={args.num_gpus}",
            str(scripts_dir / "train_deepspeed.py"),
            "--model", current_model,
            "--input", str(tested_file),
            "--output_dir", str(epoch_dir),
            "--deepspeed", args.deepspeed_config,
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--label_mode", args.label_mode,
        ]
        run_command(cmd, f"Train (epoch {epoch})")

        # Update model path for next epoch
        current_model = str(epoch_dir)

        # Cleanup tested file
        if tested_file.exists():
            tested_file.unlink()

        # Step 3c: Evaluate QA accuracy after training
        if not args.skip_qa_eval:
            cmd = [
                "torchrun", f"--nproc_per_node={args.num_gpus}",
                str(scripts_dir / "inference_ddp.py"),
                "--mode", "eval_qa",
                "--model", current_model,
                "--input", str(responses_file),
                "--output_dir", str(output_dir),
                "--batch_size", str(args.batch_size),
                "--output_file", f"eval_qa_epoch{epoch}.jsonl",
            ]
            run_command(cmd, f"Evaluate QA accuracy (epoch {epoch})")

    print(f"\n{'#'*60}")
    print("Training Complete!")
    print(f"{'#'*60}")
    print(f"Final model: {current_model}")


if __name__ == "__main__":
    main()
