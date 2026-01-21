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
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd, step_label, desc, model=None, input_file=None):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"[{step_label}] {desc}")
    if model:
        print(f"  Model: {model}")
    if input_file:
        print(f"  Input: {input_file}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: [{step_label}] {desc} failed with code {result.returncode}")
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

    # Step 0.1: Evaluate pretrained QA accuracy (collect responses, get ground truth labels)
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
        run_command(cmd, "0.1", "Evaluate pretrained QA accuracy", model=args.model)
    else:
        print(f"\n[0.1] Skipping (using existing {responses_file})")

    # Step 0.2: Evaluate pretrained judgment accuracy
    if args.start_epoch == 1:
        cmd = [
            "torchrun", f"--nproc_per_node={args.num_gpus}",
            str(scripts_dir / "inference_ddp.py"),
            "--mode", "test",
            "--model", args.model,
            "--input", str(responses_file),
            "--output_dir", str(output_dir),
            "--batch_size", str(args.batch_size),
            "--output_file", "tested_epoch0.jsonl",
            "--label_mode", args.label_mode,
        ]
        run_command(cmd, "0.2", "Evaluate pretrained judgment accuracy", model=args.model)

    # Baseline files for comparison
    # QA baseline comes from responses.jsonl (collect step already has QA results)
    baseline_qa_file = responses_file
    baseline_judgment_file = output_dir / "tested_epoch0.jsonl"

    # Training loop
    current_model = args.model

    for epoch in range(args.start_epoch, args.epochs + 1):
        print(f"\n{'#'*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'#'*60}")

        # Current epoch's tested file (from previous evaluation)
        if epoch == 1:
            tested_file = output_dir / "tested_epoch0.jsonl"
        else:
            tested_file = output_dir / f"tested_epoch{epoch-1}.jsonl"
        epoch_dir = output_dir / f"epoch_{epoch}"

        # Step N.1: Train on current tested samples
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
        run_command(cmd, f"{epoch}.1", "Train judgment", model=current_model, input_file=str(tested_file))

        # Update model path for next epoch
        current_model = str(epoch_dir)

        # Cleanup previous tested file (except epoch 0)
        if epoch > 1 and tested_file.exists():
            tested_file.unlink()

        # Step N.2: Evaluate judgment accuracy after training
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
            "--baseline", str(baseline_judgment_file),
        ]
        run_command(cmd, f"{epoch}.2", "Evaluate judgment accuracy", model=current_model)

        # Step N.3: Evaluate QA accuracy after training (detect degradation)
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
                "--baseline", str(baseline_qa_file),
            ]
            run_command(cmd, f"{epoch}.3", "Evaluate QA accuracy (detect degradation)", model=current_model)

    # Print summary
    print_summary(output_dir, args.epochs, args.skip_qa_eval)


def print_summary(output_dir: Path, epochs: int, skip_qa_eval: bool):
    """Print summary of all epochs."""
    print(f"\n{'#'*60}")
    print("Training Complete - Summary")
    print(f"{'#'*60}")

    # Collect results from all epochs
    results = []
    for epoch in range(0, epochs + 1):
        epoch_result = {"epoch": epoch}

        # Read judgment accuracy
        judgment_file = output_dir / f"tested_epoch{epoch}.jsonl"
        if judgment_file.exists():
            with open(judgment_file) as f:
                samples = [json.loads(line) for line in f]
            correct = sum(1 for s in samples if s.get("judgment_correct", False))
            epoch_result["judgment_acc"] = correct / len(samples) * 100 if samples else 0
            epoch_result["judgment_total"] = len(samples)

        # Read QA accuracy
        if not skip_qa_eval:
            qa_file = output_dir / f"eval_qa_epoch{epoch}.jsonl"
            if qa_file.exists():
                with open(qa_file) as f:
                    samples = [json.loads(line) for line in f]
                correct = sum(1 for s in samples if s.get("correct", False))
                epoch_result["qa_acc"] = correct / len(samples) * 100 if samples else 0

        results.append(epoch_result)

    # Print table
    print(f"\n{'Epoch':<8} {'Judgment Acc':<15} {'QA Acc':<12}")
    print("-" * 40)

    for r in results:
        epoch_str = f"{r['epoch']}"
        judgment_str = f"{r.get('judgment_acc', 0):.1f}%" if "judgment_acc" in r else "N/A"
        qa_str = f"{r.get('qa_acc', 0):.1f}%" if "qa_acc" in r else "N/A"
        print(f"{epoch_str:<8} {judgment_str:<15} {qa_str:<12}")

    # Print improvement
    if len(results) >= 2 and "judgment_acc" in results[0] and "judgment_acc" in results[-1]:
        judgment_diff = results[-1]["judgment_acc"] - results[0]["judgment_acc"]
        diff_str = f"+{judgment_diff:.1f}%" if judgment_diff >= 0 else f"{judgment_diff:.1f}%"
        print(f"\nJudgment improvement: {diff_str} (epoch 0 → epoch {epochs})")

    if not skip_qa_eval and len(results) >= 2 and "qa_acc" in results[0] and "qa_acc" in results[-1]:
        qa_diff = results[-1]["qa_acc"] - results[0]["qa_acc"]
        diff_str = f"+{qa_diff:.1f}%" if qa_diff >= 0 else f"{qa_diff:.1f}%"
        print(f"QA improvement: {diff_str} (epoch 0 → epoch {epochs})")

    print(f"\nFinal model: {output_dir / f'epoch_{epochs}'}")


if __name__ == "__main__":
    main()
