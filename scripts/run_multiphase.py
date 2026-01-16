"""
Run Multi-phase Pipeline

Unified entry point for the three-phase training workflow:
- Phase 1: Initial judgment training (existing pipeline)
- Phase 2: Knowledge learning (new)
- Phase 3: Update judgment with knowledge (new)

Usage:
    # Run all phases (experiment name auto-generated)
    python run_multiphase.py --model Qwen/Qwen2.5-0.5B-Instruct --num_samples 1000

    # Run specific phase
    python run_multiphase.py --phase 2

    # Resume from checkpoint
    python run_multiphase.py --experiment <name> --resume
"""

import argparse
import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import MultiPhasePipeline, load_experiment
from src.multi_gpu_inference import create_inference
from src.evaluator import is_correct
from src.data_loader import load_triviaqa
from tqdm import tqdm


def test_qa_accuracy(
    model_path: str,
    split: str = "train",
    num_samples: int = 100,
    num_trials: int = 5,
    inference_batch_size: int = 16,
    multi_gpu: bool = False,
    num_gpus: int = None,
) -> dict:
    """
    Test QA accuracy on a dataset split.

    Returns:
        dict with accuracy, correct count, and total count
    """
    print(f"  Testing QA accuracy on {split} split ({num_samples} samples)...")

    samples = load_triviaqa(split=split, num_samples=num_samples)

    inference = create_inference(
        model_name=model_path,
        inference_batch_size=inference_batch_size,
        temperature=1.0,
        multi_gpu=multi_gpu,
        num_gpus=num_gpus,
    )

    correct_count = 0
    total = 0

    for sample in tqdm(samples, desc=f"QA test ({split})", leave=False):
        question = sample["question"]
        gold_answers = sample.get("normalized_answers", sample.get("answers", []))

        if not gold_answers:
            continue

        # Generate responses
        responses = inference.generate(
            f"Question: {question}\nAnswer:",
            num_samples=num_trials
        )

        # Check if any response is correct
        any_correct = any(is_correct(r, gold_answers) for r in responses)
        if any_correct:
            correct_count += 1
        total += 1

    accuracy = correct_count / total if total > 0 else 0

    # Clean up
    if hasattr(inference, 'shutdown'):
        inference.shutdown()
    del inference
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "qa_accuracy": accuracy * 100,  # As percentage
        "qa_correct": correct_count,
        "qa_total": total,
    }


def parse_eval_metrics(output: str) -> dict:
    """Parse evaluation output to extract key metrics."""
    metrics = {}

    # Extract exact match rate
    match = re.search(r'Exact match.*?(\d+\.?\d*)%', output)
    if match:
        metrics['exact_match_rate'] = float(match.group(1))

    # Extract predicted distribution
    match = re.search(r'Predicted distribution: can=(\d+), uncertain=(\d+), cannot=(\d+)', output)
    if match:
        metrics['pred_can'] = int(match.group(1))
        metrics['pred_uncertain'] = int(match.group(2))
        metrics['pred_cannot'] = int(match.group(3))

    # Extract actual distribution
    match = re.search(r'Actual distribution:\s+can=(\d+), uncertain=(\d+), cannot=(\d+)', output)
    if match:
        metrics['actual_can'] = int(match.group(1))
        metrics['actual_uncertain'] = int(match.group(2))
        metrics['actual_cannot'] = int(match.group(3))

    # Extract confusion matrix
    # predicted_can row
    match = re.search(r'predicted_can\s+(\d+)\s+(\d+)\s+(\d+)', output)
    if match:
        metrics['cm_can_can'] = int(match.group(1))
        metrics['cm_can_uncertain'] = int(match.group(2))
        metrics['cm_can_cannot'] = int(match.group(3))

    # predicted_uncertain row
    match = re.search(r'predicted_uncertain\s+(\d+)\s+(\d+)\s+(\d+)', output)
    if match:
        metrics['cm_uncertain_can'] = int(match.group(1))
        metrics['cm_uncertain_uncertain'] = int(match.group(2))
        metrics['cm_uncertain_cannot'] = int(match.group(3))

    # predicted_cannot row
    match = re.search(r'predicted_cannot\s+(\d+)\s+(\d+)\s+(\d+)', output)
    if match:
        metrics['cm_cannot_can'] = int(match.group(1))
        metrics['cm_cannot_uncertain'] = int(match.group(2))
        metrics['cm_cannot_cannot'] = int(match.group(3))

    return metrics


def print_confusion_matrix(name: str, metrics: dict):
    """Print a single confusion matrix."""
    print(f"\n  {name}")
    print(f"  {'':20} {'actual_can':>12} {'actual_unc':>12} {'actual_cannot':>14}")
    print(f"  predicted_can      {metrics.get('cm_can_can', 0):>12} {metrics.get('cm_can_uncertain', 0):>12} {metrics.get('cm_can_cannot', 0):>14}")
    print(f"  predicted_uncertain{metrics.get('cm_uncertain_can', 0):>12} {metrics.get('cm_uncertain_uncertain', 0):>12} {metrics.get('cm_uncertain_cannot', 0):>14}")
    print(f"  predicted_cannot   {metrics.get('cm_cannot_can', 0):>12} {metrics.get('cm_cannot_uncertain', 0):>12} {metrics.get('cm_cannot_cannot', 0):>14}")


def print_phase_summary(title: str, results: dict):
    """Print a summary table for phase results."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

    # Exact match rates summary
    print(f"\n  EXACT MATCH ACCURACY")
    print(f"  {'':20} {'TRAIN':>20} {'VALIDATION':>20}")
    print("  " + "-" * 66)

    before_train = results.get('before_train', {}).get('exact_match_rate', 0)
    before_val = results.get('before_val', {}).get('exact_match_rate', 0)
    after_train = results.get('after_train', {}).get('exact_match_rate', 0)
    after_val = results.get('after_val', {}).get('exact_match_rate', 0)

    print(f"  {'Before training':20} {before_train:>19.1f}% {before_val:>19.1f}%")
    print(f"  {'After training':20} {after_train:>19.1f}% {after_val:>19.1f}%")
    print("  " + "-" * 66)

    train_imp = after_train - before_train
    val_imp = after_val - before_val
    print(f"  {'Improvement':20} {train_imp:>+18.1f}% {val_imp:>+18.1f}%")

    # Confusion matrices
    print("\n" + "=" * 70)
    print("  CONFUSION MATRICES")
    print("=" * 70)

    print_confusion_matrix("Before Training - TRAIN", results.get('before_train', {}))
    print_confusion_matrix("Before Training - VALIDATION", results.get('before_val', {}))
    print_confusion_matrix("After Training - TRAIN", results.get('after_train', {}))
    print_confusion_matrix("After Training - VALIDATION", results.get('after_val', {}))

    print("\n" + "=" * 70)


def generate_experiment_name(model: str, dataset: str, train_samples: int, test_samples: int) -> str:
    """Generate experiment name from parameters."""
    # Extract model short name (e.g., "Qwen/Qwen2.5-0.5B-Instruct" -> "Qwen2.5-0.5B")
    model_short = model.split("/")[-1].replace("-Instruct", "")

    # Dataset short name
    dataset_short = dataset.replace("/", "_")

    # Timestamp for uniqueness
    timestamp = datetime.now().strftime("%m%d_%H%M")

    return f"{model_short}_{dataset_short}_train{train_samples}_test{test_samples}_{timestamp}"


def save_config_log(output_dir: Path, args, experiment_name: str):
    """Save configuration to log file."""
    config = {
        "experiment_name": experiment_name,
        "created_at": datetime.now().isoformat(),
        "model": args.model,
        "dataset": args.dataset,
        "train_samples": args.num_samples,
        "test_samples": args.test_samples,
        "num_trials": args.num_trials,
        "epochs": args.epochs,
        "knowledge_epochs": args.knowledge_epochs,
        "batch_size": args.batch_size,
        "inference_batch_size": args.inference_batch_size,
        "learning_rate": args.lr,
        "no_lora": args.no_lora,
        "adaptive": args.adaptive,
        "max_steps_per_sample": args.max_steps_per_sample,
        "multi_gpu": args.multi_gpu,
        "num_gpus": args.num_gpus,
    }

    log_path = output_dir / "config.json"
    with open(log_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Config saved to: {log_path}")
    return config


def is_phase_completed(phase: int, pipeline: MultiPhasePipeline, args) -> bool:
    """Check if a phase has already been completed."""
    if phase == 1:
        phase_output = pipeline.get_phase_output_dir("phase1_judgment")
        # Check if judgment_v1 model exists
        if args.no_lora:
            marker = phase_output / "judgment_v1" / "config.json"
        else:
            marker = phase_output / "judgment_v1" / "adapter_config.json"
        return marker.exists()

    elif phase == 2:
        phase_output = pipeline.get_phase_output_dir("phase2_knowledge")
        # Check if knowledge model exists (full fine-tuning or merged)
        if args.no_lora:
            marker = phase_output / "knowledge" / "config.json"
        else:
            marker = phase_output / "base_with_knowledge" / "config.json"
        return marker.exists()

    elif phase == 3:
        phase_output = pipeline.get_phase_output_dir("phase3_judgment")
        # Check if judgment_v2 model exists
        if args.no_lora:
            marker = phase_output / "judgment_v2" / "config.json"
        else:
            marker = phase_output / "judgment_v2" / "adapter_config.json"
        return marker.exists()

    return False


def run_phase1(args, pipeline: MultiPhasePipeline):
    """Run Phase 1: Initial judgment training (existing steps 1-4)."""
    print("\n" + "=" * 60)
    print("Phase 1: Initial Judgment Training")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    phase_output = pipeline.get_phase_output_dir("phase1_judgment")

    # Store evaluation results for summary
    eval_results = {}

    # Step 1: Collect responses (using train split)
    print("\n[Step 1.1] Collecting responses from train split...")
    cmd = [
        sys.executable, str(project_root / "scripts/step1_collect_responses.py"),
        "--model", args.model,
        "--num_samples", str(args.num_samples),
        "--output", str(phase_output / "responses.jsonl"),
        "--num_trials", str(args.num_trials),
        "--inference_batch_size", str(args.inference_batch_size),
        "--split", "train",
    ]
    if args.multi_gpu:
        cmd.append("--multi_gpu")
    if args.num_gpus is not None:
        cmd.extend(["--num_gpus", str(args.num_gpus)])
    subprocess.run(cmd, check=True)

    # Step 2: Build dataset
    print("\n[Step 1.2] Building judgment training data...")
    cmd = [
        sys.executable, str(project_root / "scripts/step2_build_dataset.py"),
        "--input", str(phase_output / "responses.jsonl"),
        "--output", str(phase_output / "training_data.jsonl"),
    ]
    subprocess.run(cmd, check=True)

    # Step 3: Evaluate BEFORE training (baseline) - both Judgment and QA
    print("\n[Step 1.3] Evaluating BASELINE (before training)...")

    # Judgment evaluation
    print("\n[Step 1.3a] Baseline JUDGMENT on TRAIN split...")
    cmd = [
        sys.executable, str(project_root / "scripts/step4_evaluate.py"),
        "--model", args.model,
        "--lora_path", "none",  # No LoRA - baseline
        "--num_samples", str(args.num_samples),  # Use train samples (same as training)
        "--num_trials", str(args.num_trials),
        "--inference_batch_size", str(args.inference_batch_size),
        "--split", "train",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    eval_results['before_train'] = parse_eval_metrics(result.stdout)

    print("\n[Step 1.3b] Baseline JUDGMENT on VALIDATION split...")
    cmd = [
        sys.executable, str(project_root / "scripts/step4_evaluate.py"),
        "--model", args.model,
        "--lora_path", "none",  # No LoRA - baseline
        "--num_samples", str(args.test_samples),
        "--num_trials", str(args.num_trials),
        "--inference_batch_size", str(args.inference_batch_size),
        "--split", "validation",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    eval_results['before_val'] = parse_eval_metrics(result.stdout)

    # QA evaluation (baseline)
    print("\n[Step 1.3c] Baseline QA accuracy...")
    qa_before_train = test_qa_accuracy(
        args.model, split="train", num_samples=args.test_samples,
        num_trials=args.num_trials, inference_batch_size=args.inference_batch_size,
        multi_gpu=args.multi_gpu, num_gpus=args.num_gpus
    )
    eval_results['before_train'].update(qa_before_train)
    print(f"  Train QA: {qa_before_train['qa_accuracy']:.1f}%")

    qa_before_val = test_qa_accuracy(
        args.model, split="validation", num_samples=args.test_samples,
        num_trials=args.num_trials, inference_batch_size=args.inference_batch_size,
        multi_gpu=args.multi_gpu, num_gpus=args.num_gpus
    )
    eval_results['before_val'].update(qa_before_val)
    print(f"  Validation QA: {qa_before_val['qa_accuracy']:.1f}%")

    # Step 4: Train judgment
    print("\n[Step 1.4] Training judgment ability...")
    cmd = [
        sys.executable, str(project_root / "scripts/step3_train.py"),
        "--model", args.model,
        "--input", str(phase_output / "training_data.jsonl"),
        "--output_dir", str(phase_output / "judgment_v1"),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--max_steps_per_sample", str(args.max_steps_per_sample),
        "--skip_correct",  # Skip samples already judged correctly
    ]
    if args.no_lora:
        cmd.append("--no_lora")
    if args.adaptive:
        cmd.append("--adaptive")
    subprocess.run(cmd, check=True)

    # Step 5: Evaluate AFTER training on both splits
    print("\n[Step 1.5] Evaluating AFTER training...")

    # For full fine-tuning, use trained model directly; for LoRA, use base + adapter
    if args.no_lora:
        eval_model = str(phase_output / "judgment_v1")  # Full model saved here
        eval_lora = "none"
    else:
        eval_model = args.model
        eval_lora = str(phase_output / "judgment_v1")

    print("\n[Step 1.5a] After training on TRAIN split...")
    cmd = [
        sys.executable, str(project_root / "scripts/step4_evaluate.py"),
        "--model", eval_model,
        "--lora_path", eval_lora,
        "--num_samples", str(args.num_samples),  # Use train samples (same as training)
        "--num_trials", str(args.num_trials),
        "--inference_batch_size", str(args.inference_batch_size),
        "--split", "train",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    eval_results['after_train'] = parse_eval_metrics(result.stdout)

    print("\n[Step 1.5b] After training JUDGMENT on VALIDATION split...")
    cmd = [
        sys.executable, str(project_root / "scripts/step4_evaluate.py"),
        "--model", eval_model,
        "--lora_path", eval_lora,
        "--num_samples", str(args.test_samples),
        "--num_trials", str(args.num_trials),
        "--inference_batch_size", str(args.inference_batch_size),
        "--split", "validation",  # Test generalization
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    eval_results['after_val'] = parse_eval_metrics(result.stdout)

    # QA evaluation after training (to verify judgment training didn't hurt QA)
    print("\n[Step 1.5c] After training QA accuracy...")
    # For QA test, use the trained model path
    qa_model = eval_model if args.no_lora else str(phase_output / "judgment_v1")
    # Note: For LoRA, we need to load base + adapter for QA test
    # But the simple approach is to test with base model since judgment LoRA
    # shouldn't affect QA much. For full fine-tuning, use trained model.
    if args.no_lora:
        qa_test_model = eval_model
    else:
        # For LoRA judgment training, QA ability should be same as base model
        # since we only trained judgment task. But let's test the merged model
        # if available, otherwise test base model.
        qa_test_model = args.model  # Base model (judgment LoRA shouldn't affect QA)

    qa_after_train = test_qa_accuracy(
        qa_test_model, split="train", num_samples=args.test_samples,
        num_trials=args.num_trials, inference_batch_size=args.inference_batch_size,
        multi_gpu=args.multi_gpu, num_gpus=args.num_gpus
    )
    eval_results['after_train'].update(qa_after_train)
    print(f"  Train QA: {qa_after_train['qa_accuracy']:.1f}%")

    qa_after_val = test_qa_accuracy(
        qa_test_model, split="validation", num_samples=args.test_samples,
        num_trials=args.num_trials, inference_batch_size=args.inference_batch_size,
        multi_gpu=args.multi_gpu, num_gpus=args.num_gpus
    )
    eval_results['after_val'].update(qa_after_val)
    print(f"  Validation QA: {qa_after_val['qa_accuracy']:.1f}%")

    # Print summary table
    print_phase_summary("PHASE 1 SUMMARY: Judgment Training", eval_results)

    # Record phase result
    pipeline.record_phase_result(
        phase_name="phase1_judgment",
        status="completed",
        metrics=eval_results,
        output_paths={
            "responses": str(phase_output / "responses.jsonl"),
            "training_data": str(phase_output / "training_data.jsonl"),
            "judgment_v1": str(phase_output / "judgment_v1"),
        }
    )
    pipeline.state.current_phase = 1
    pipeline._save_state()

    print("\nPhase 1 completed!")


def run_phase2(args, pipeline: MultiPhasePipeline):
    """Run Phase 2: Knowledge learning."""
    print("\n" + "=" * 60)
    print("Phase 2: Knowledge Learning")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    phase1_output = pipeline.get_phase_output_dir("phase1_judgment")
    phase2_output = pipeline.get_phase_output_dir("phase2_knowledge")

    # Use Phase 1 responses as input
    input_path = phase1_output / "responses.jsonl"
    if not input_path.exists():
        # Fallback to default location
        input_path = project_root / "data/step1_responses.jsonl"

    cmd = [
        sys.executable, str(project_root / "scripts/phase2_knowledge.py"),
        "--model", args.model,
        "--input", str(input_path),
        "--output_dir", str(phase2_output),
        "--epochs", str(args.knowledge_epochs),
        "--batch_size", str(args.batch_size),
        "--inference_batch_size", str(args.inference_batch_size),
        "--test_samples", str(args.test_samples),
        "--lr", str(args.lr),
        "--max_steps_per_sample", str(args.max_steps_per_sample),
        # Only train samples the model doesn't already know
        "--filter_ability", "cannot", "uncertain",
        "--skip_correct",
    ]
    if args.no_lora:
        cmd.append("--no_lora")
    if args.adaptive:
        cmd.append("--adaptive")
    if args.multi_gpu:
        cmd.append("--multi_gpu")
    if args.num_gpus is not None:
        cmd.extend(["--num_gpus", str(args.num_gpus)])
    subprocess.run(cmd, check=True)

    pipeline.state.current_phase = 2
    pipeline._save_state()

    print("\nPhase 2 completed!")


def run_phase3(args, pipeline: MultiPhasePipeline):
    """Run Phase 3: Update judgment with knowledge."""
    print("\n" + "=" * 60)
    print("Phase 3: Update Judgment with Knowledge")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    phase1_output = pipeline.get_phase_output_dir("phase1_judgment")
    phase2_output = pipeline.get_phase_output_dir("phase2_knowledge")
    phase3_output = pipeline.get_phase_output_dir("phase3_judgment")

    # Use knowledge model from Phase 2
    # For LoRA: merged model at base_with_knowledge
    # For full fine-tuning: model at knowledge
    if args.no_lora:
        # Full fine-tuning: model saved directly
        base_model = phase2_output / "knowledge"
    else:
        # LoRA: use merged model
        base_model = phase2_output / "base_with_knowledge"

    if not base_model.exists():
        print(f"Warning: Knowledge model not found at {base_model}")
        print(f"Using original model: {args.model}")
        base_model = args.model
    else:
        print(f"Using knowledge model from: {base_model}")

    # Use Phase 1 responses as input
    input_path = phase1_output / "responses.jsonl"
    if not input_path.exists():
        input_path = project_root / "data/step1_responses.jsonl"

    cmd = [
        sys.executable, str(project_root / "scripts/phase3_update_judgment.py"),
        "--base_model", str(base_model),
        "--original_base", args.model,
        "--input", str(input_path),
        "--output_dir", str(phase3_output),
        "--num_samples", str(args.num_samples),
        "--num_trials", str(args.num_trials),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--inference_batch_size", str(args.inference_batch_size),
        "--test_samples", str(args.test_samples),
        "--lr", str(args.lr),
        "--max_steps_per_sample", str(args.max_steps_per_sample),
        "--skip_correct",  # Skip samples already judged correctly
    ]
    if args.no_lora:
        cmd.append("--no_lora")
    if args.adaptive:
        cmd.append("--adaptive")
    if args.multi_gpu:
        cmd.append("--multi_gpu")
    if args.num_gpus is not None:
        cmd.extend(["--num_gpus", str(args.num_gpus)])
    subprocess.run(cmd, check=True)

    pipeline.state.current_phase = 3
    pipeline._save_state()

    print("\nPhase 3 completed!")


def main():
    parser = argparse.ArgumentParser(description="Run Multi-phase Pipeline")

    # Experiment (optional - auto-generated if not provided)
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name (auto-generated if not provided)")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="Base output directory")

    # Model and dataset
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", type=str, default="triviaqa",
                        help="Dataset name (for experiment naming)")

    # Phase selection
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                        help="Run specific phase (default: run all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if phase already completed")
    parser.add_argument("--summary", action="store_true",
                        help="Only print summary of existing experiment (no training)")

    # Data params
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Training samples")
    parser.add_argument("--test_samples", type=int, default=100,
                        help="Test samples for evaluation")
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--inference_batch_size", type=int, default=16)

    # Training params
    parser.add_argument("--epochs", type=int, default=2,
                        help="Epochs for judgment training (default: 2 for adaptive)")
    parser.add_argument("--knowledge_epochs", type=int, default=2,
                        help="Epochs for knowledge training (default: 2 for adaptive)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (1e-4 for LoRA, 1e-5 for full fine-tuning)")
    parser.add_argument("--no_lora", action="store_true",
                        help="Disable LoRA for full fine-tuning")
    parser.add_argument("--adaptive", action="store_true", default=True,
                        help="Use adaptive training (train each sample until learned)")
    parser.add_argument("--max_steps_per_sample", type=int, default=10,
                        help="Max training steps per sample in adaptive mode")

    # Multi-GPU params
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use multi-GPU inference (one model per GPU)")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs to use (default: all available)")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_base = project_root / args.output_dir

    # Handle --summary mode: just print existing experiment summary
    if args.summary:
        if args.experiment is None:
            print("Error: --summary requires --experiment <name> to specify which experiment to summarize")
            print("\nAvailable experiments:")
            if output_base.exists():
                for exp_dir in sorted(output_base.iterdir()):
                    if exp_dir.is_dir() and (exp_dir / "pipeline_state.json").exists():
                        print(f"  - {exp_dir.name}")
            return

        try:
            pipeline = load_experiment(args.experiment, str(output_base))
            pipeline.print_summary()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nAvailable experiments:")
            if output_base.exists():
                for exp_dir in sorted(output_base.iterdir()):
                    if exp_dir.is_dir() and (exp_dir / "pipeline_state.json").exists():
                        print(f"  - {exp_dir.name}")
        return

    # Auto-generate experiment name if not provided
    if args.experiment is None:
        experiment_name = generate_experiment_name(
            model=args.model,
            dataset=args.dataset,
            train_samples=args.num_samples,
            test_samples=args.test_samples
        )
    else:
        experiment_name = args.experiment

    # Create experiment directory
    experiment_dir = output_base / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save config to log file
    save_config_log(experiment_dir, args, experiment_name)

    # Create or load pipeline
    pipeline = MultiPhasePipeline(
        experiment_name=experiment_name,
        base_model=args.model,
        output_dir=str(output_base),
        config=vars(args)
    )

    print("=" * 60)
    print(f"Multi-phase Pipeline")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Train samples: {args.num_samples}")
    print(f"Test samples: {args.test_samples}")
    print(f"Output: {pipeline.output_dir}")
    print(f"Current phase: {pipeline.state.current_phase}")
    print("=" * 60)

    # Determine which phases to run
    if args.phase:
        # Run specific phase
        phases_to_run = [args.phase]
    elif args.resume:
        # Resume from checkpoint
        start_phase = pipeline.state.current_phase + 1
        phases_to_run = list(range(start_phase, 4))
        if not phases_to_run:
            print("All phases already completed!")
            pipeline.print_summary()
            return
    else:
        # Run all phases
        phases_to_run = [1, 2, 3]

    print(f"Phases to run: {phases_to_run}")
    if args.force:
        print("Force mode: will re-run all specified phases")

    # Execute phases
    for phase in phases_to_run:
        # Check if phase already completed (unless --force)
        if not args.force and is_phase_completed(phase, pipeline, args):
            print(f"\n{'=' * 60}")
            print(f"Phase {phase} already completed, skipping...")
            print(f"(Use --force to re-run)")
            print(f"{'=' * 60}")
            continue

        if phase == 1:
            run_phase1(args, pipeline)
        elif phase == 2:
            run_phase2(args, pipeline)
        elif phase == 3:
            run_phase3(args, pipeline)

    # Final summary
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    pipeline.print_summary()


if __name__ == "__main__":
    main()
