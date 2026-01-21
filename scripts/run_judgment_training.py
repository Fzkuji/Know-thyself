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
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-generated if not specified)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--num_val_samples", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--label_mode", type=str, default="binary", choices=["binary", "uncertainty"])
    parser.add_argument("--deepspeed_config", type=str, default="configs/ds_config_zero3.json")

    # Resume options
    parser.add_argument("--skip_collect", action="store_true", help="Skip response collection (use existing)")
    parser.add_argument("--start_epoch", type=int, default=1, help="Start from this epoch")

    args = parser.parse_args()

    # Auto-generate output_dir if not specified
    if args.output_dir is None:
        from datetime import datetime
        model_name = args.model.split("/")[-1].lower().replace("-", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lr_str = f"{args.lr:.0e}".replace("-", "").replace("+", "")
        args.output_dir = f"experiments/{model_name}_{args.label_mode}_lr{lr_str}_{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).parent
    responses_file = output_dir / "responses.jsonl"
    val_responses_file = output_dir / "val_responses.jsonl"

    print(f"\n{'#'*60}")
    print("Judgment Training Pipeline")
    print(f"{'#'*60}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Label mode: {args.label_mode}")
    print(f"Train samples: {args.num_samples}")
    print(f"Val samples: {args.num_val_samples}")

    # =========================================================================
    # Step 0: Evaluate pretrained model on both train and validation sets
    # =========================================================================

    # Step 0.1: Evaluate pretrained QA accuracy (Train)
    if not args.skip_collect and not responses_file.exists():
        cmd = [
            "torchrun", f"--nproc_per_node={args.num_gpus}",
            str(scripts_dir / "inference_ddp.py"),
            "--mode", "collect",
            "--model", args.model,
            "--output_dir", str(output_dir),
            "--num_samples", str(args.num_samples),
            "--split", "train",
            "--batch_size", str(args.batch_size),
            "--label_mode", args.label_mode,
        ]
        run_command(cmd, "0.1", "Evaluate pretrained QA accuracy (Train)", model=args.model)
    else:
        print(f"\n[0.1] Skipping (using existing {responses_file})")

    # Step 0.2: Evaluate pretrained judgment accuracy (Train)
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
        run_command(cmd, "0.2", "Evaluate pretrained judgment accuracy (Train)", model=args.model)

    # Step 0.3: Evaluate pretrained QA accuracy (Validation)
    if not args.skip_collect and not val_responses_file.exists():
        cmd = [
            "torchrun", f"--nproc_per_node={args.num_gpus}",
            str(scripts_dir / "inference_ddp.py"),
            "--mode", "collect",
            "--model", args.model,
            "--output_dir", str(output_dir),
            "--num_samples", str(args.num_val_samples),
            "--split", "validation",
            "--batch_size", str(args.batch_size),
            "--label_mode", args.label_mode,
            "--output_file", "val_responses.jsonl",
        ]
        run_command(cmd, "0.3", "Evaluate pretrained QA accuracy (Val)", model=args.model)
    else:
        print(f"\n[0.3] Skipping (using existing {val_responses_file})")

    # Step 0.4: Evaluate pretrained judgment accuracy (Validation)
    if args.start_epoch == 1:
        cmd = [
            "torchrun", f"--nproc_per_node={args.num_gpus}",
            str(scripts_dir / "inference_ddp.py"),
            "--mode", "test",
            "--model", args.model,
            "--input", str(val_responses_file),
            "--output_dir", str(output_dir),
            "--batch_size", str(args.batch_size),
            "--output_file", "val_tested_epoch0.jsonl",
            "--label_mode", args.label_mode,
        ]
        run_command(cmd, "0.4", "Evaluate pretrained judgment accuracy (Val)", model=args.model)

    # Baseline files for comparison
    baseline_qa_file = responses_file
    baseline_judgment_file = output_dir / "tested_epoch0.jsonl"
    baseline_val_qa_file = val_responses_file
    baseline_val_judgment_file = output_dir / "val_tested_epoch0.jsonl"

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

        # Note: Keep tested files for summary statistics
        # Previously deleted to save space, but needed for print_summary

        # Step N.2: Evaluate QA accuracy after training (get new ground truth)
        cmd = [
            "torchrun", f"--nproc_per_node={args.num_gpus}",
            str(scripts_dir / "inference_ddp.py"),
            "--mode", "collect",
            "--model", current_model,
            "--input", str(responses_file),
            "--output_dir", str(output_dir),
            "--batch_size", str(args.batch_size),
            "--output_file", f"responses_epoch{epoch}.jsonl",
            "--label_mode", args.label_mode,
            "--baseline", str(baseline_qa_file),
        ]
        run_command(cmd, f"{epoch}.2", "Evaluate SFT QA accuracy", model=current_model)

        # Step N.3: Evaluate judgment accuracy after training (Train)
        responses_epoch_file = output_dir / f"responses_epoch{epoch}.jsonl"
        cmd = [
            "torchrun", f"--nproc_per_node={args.num_gpus}",
            str(scripts_dir / "inference_ddp.py"),
            "--mode", "test",
            "--model", current_model,
            "--input", str(responses_epoch_file),
            "--output_dir", str(output_dir),
            "--batch_size", str(args.batch_size),
            "--output_file", f"tested_epoch{epoch}.jsonl",
            "--label_mode", args.label_mode,
            "--baseline", str(baseline_judgment_file),
        ]
        run_command(cmd, f"{epoch}.3", "Evaluate SFT judgment accuracy (Train)", model=current_model)

        # Step N.4: Evaluate QA accuracy after training (Validation)
        cmd = [
            "torchrun", f"--nproc_per_node={args.num_gpus}",
            str(scripts_dir / "inference_ddp.py"),
            "--mode", "collect",
            "--model", current_model,
            "--input", str(val_responses_file),
            "--output_dir", str(output_dir),
            "--batch_size", str(args.batch_size),
            "--output_file", f"val_responses_epoch{epoch}.jsonl",
            "--label_mode", args.label_mode,
            "--baseline", str(baseline_val_qa_file),
        ]
        run_command(cmd, f"{epoch}.4", "Evaluate SFT QA accuracy (Val)", model=current_model)

        # Step N.5: Evaluate judgment accuracy after training (Validation)
        val_responses_epoch_file = output_dir / f"val_responses_epoch{epoch}.jsonl"
        cmd = [
            "torchrun", f"--nproc_per_node={args.num_gpus}",
            str(scripts_dir / "inference_ddp.py"),
            "--mode", "test",
            "--model", current_model,
            "--input", str(val_responses_epoch_file),
            "--output_dir", str(output_dir),
            "--batch_size", str(args.batch_size),
            "--output_file", f"val_tested_epoch{epoch}.jsonl",
            "--label_mode", args.label_mode,
            "--baseline", str(baseline_val_judgment_file),
        ]
        run_command(cmd, f"{epoch}.5", "Evaluate SFT judgment accuracy (Val)", model=current_model)

    # Print summary
    print_summary(output_dir, args.epochs)


def compute_auroc(samples):
    """Compute AUROC from samples with yes_prob field."""
    from sklearn.metrics import roc_auc_score

    # Filter samples with yes_prob
    valid_samples = [s for s in samples if "yes_prob" in s]
    if not valid_samples:
        return None

    y_true = [1 if s["ability"] == "can" else 0 for s in valid_samples]
    y_scores = [s["yes_prob"] for s in valid_samples]

    # Need both classes for AUROC
    if len(set(y_true)) < 2:
        return None

    return roc_auc_score(y_true, y_scores)


def print_summary(output_dir: Path, epochs: int):
    """Print summary of all epochs with train and validation results."""
    print(f"\n{'#'*60}")
    print("Training Complete - Summary")
    print(f"{'#'*60}")

    # Collect results from all epochs
    results = []
    for epoch in range(0, epochs + 1):
        epoch_result = {"epoch": epoch}

        # Read Train judgment accuracy and AUROC from tested_epochN.jsonl
        judgment_file = output_dir / f"tested_epoch{epoch}.jsonl"
        if judgment_file.exists():
            with open(judgment_file) as f:
                samples = [json.loads(line) for line in f]
            correct = sum(1 for s in samples if s.get("judgment_correct", False))
            epoch_result["train_judgment_acc"] = correct / len(samples) * 100 if samples else 0
            # Compute AUROC
            auroc = compute_auroc(samples)
            if auroc is not None:
                epoch_result["train_auroc"] = auroc

        # Read Train QA accuracy from responses.jsonl (epoch 0) or responses_epochN.jsonl (epoch N)
        if epoch == 0:
            qa_file = output_dir / "responses.jsonl"
        else:
            qa_file = output_dir / f"responses_epoch{epoch}.jsonl"
        if qa_file.exists():
            with open(qa_file) as f:
                samples = [json.loads(line) for line in f]
            correct = sum(1 for s in samples if s.get("ability") == "can")
            epoch_result["train_qa_acc"] = correct / len(samples) * 100 if samples else 0

        # Read Validation judgment accuracy and AUROC from val_tested_epochN.jsonl
        val_judgment_file = output_dir / f"val_tested_epoch{epoch}.jsonl"
        if val_judgment_file.exists():
            with open(val_judgment_file) as f:
                samples = [json.loads(line) for line in f]
            correct = sum(1 for s in samples if s.get("judgment_correct", False))
            epoch_result["val_judgment_acc"] = correct / len(samples) * 100 if samples else 0
            # Compute AUROC
            auroc = compute_auroc(samples)
            if auroc is not None:
                epoch_result["val_auroc"] = auroc

        # Read Validation QA accuracy from val_responses.jsonl (epoch 0) or val_responses_epochN.jsonl (epoch N)
        if epoch == 0:
            val_qa_file = output_dir / "val_responses.jsonl"
        else:
            val_qa_file = output_dir / f"val_responses_epoch{epoch}.jsonl"
        if val_qa_file.exists():
            with open(val_qa_file) as f:
                samples = [json.loads(line) for line in f]
            correct = sum(1 for s in samples if s.get("ability") == "can")
            epoch_result["val_qa_acc"] = correct / len(samples) * 100 if samples else 0

        results.append(epoch_result)

    # Print table with 6 columns (including AUROC)
    print(f"\n{'Epoch':<8} {'Train Judg':<12} {'Train QA':<12} {'Train AUROC':<12} {'Val Judg':<12} {'Val QA':<12} {'Val AUROC':<12}")
    print("-" * 84)

    for r in results:
        epoch_str = f"{r['epoch']}"
        train_judg = f"{r.get('train_judgment_acc', 0):.1f}%" if "train_judgment_acc" in r else "N/A"
        train_qa = f"{r.get('train_qa_acc', 0):.1f}%" if "train_qa_acc" in r else "N/A"
        train_auroc = f"{r.get('train_auroc', 0):.4f}" if "train_auroc" in r else "N/A"
        val_judg = f"{r.get('val_judgment_acc', 0):.1f}%" if "val_judgment_acc" in r else "N/A"
        val_qa = f"{r.get('val_qa_acc', 0):.1f}%" if "val_qa_acc" in r else "N/A"
        val_auroc = f"{r.get('val_auroc', 0):.4f}" if "val_auroc" in r else "N/A"
        print(f"{epoch_str:<8} {train_judg:<12} {train_qa:<12} {train_auroc:<12} {val_judg:<12} {val_qa:<12} {val_auroc:<12}")

    # Print improvement summary
    print(f"\n{'='*84}")
    print("Improvement (epoch 0 → epoch {})".format(epochs))
    print("="*84)

    if len(results) >= 2:
        for metric, label, is_pct in [
            ("train_judgment_acc", "Train Judgment", True),
            ("train_qa_acc", "Train QA", True),
            ("train_auroc", "Train AUROC", False),
            ("val_judgment_acc", "Val Judgment", True),
            ("val_qa_acc", "Val QA", True),
            ("val_auroc", "Val AUROC", False),
        ]:
            if metric in results[0] and metric in results[-1]:
                diff = results[-1][metric] - results[0][metric]
                if is_pct:
                    diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
                else:
                    diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
                print(f"  {label:<16}: {diff_str}")

    print(f"\nFinal model: {output_dir / f'epoch_{epochs}'}")


if __name__ == "__main__":
    main()
