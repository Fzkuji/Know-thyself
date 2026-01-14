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
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import MultiPhasePipeline, load_experiment


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
    }

    log_path = output_dir / "config.json"
    with open(log_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Config saved to: {log_path}")
    return config


def run_phase1(args, pipeline: MultiPhasePipeline):
    """Run Phase 1: Initial judgment training (existing steps 1-4)."""
    print("\n" + "=" * 60)
    print("Phase 1: Initial Judgment Training")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    phase_output = pipeline.get_phase_output_dir("phase1_judgment")

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
    subprocess.run(cmd, check=True)

    # Step 2: Build dataset
    print("\n[Step 1.2] Building judgment training data...")
    cmd = [
        sys.executable, str(project_root / "scripts/step2_build_dataset.py"),
        "--input", str(phase_output / "responses.jsonl"),
        "--output", str(phase_output / "training_data.jsonl"),
    ]
    subprocess.run(cmd, check=True)

    # Step 3: Train judgment
    print("\n[Step 1.3] Training judgment ability...")
    cmd = [
        sys.executable, str(project_root / "scripts/step3_train.py"),
        "--model", args.model,
        "--input", str(phase_output / "training_data.jsonl"),
        "--output_dir", str(phase_output / "lora_judgment_v1"),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
    ]
    subprocess.run(cmd, check=True)

    # Step 4: Evaluate on both train and validation splits
    print("\n[Step 1.4a] Evaluating judgment accuracy on TRAIN split...")
    cmd = [
        sys.executable, str(project_root / "scripts/step4_evaluate.py"),
        "--model", args.model,
        "--lora_path", str(phase_output / "lora_judgment_v1"),
        "--num_samples", str(args.test_samples),
        "--num_trials", str(args.num_trials),
        "--inference_batch_size", str(args.inference_batch_size),
        "--split", "train",  # Verify model learned training data
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    print("\n[Step 1.4b] Evaluating judgment accuracy on VALIDATION split...")
    cmd = [
        sys.executable, str(project_root / "scripts/step4_evaluate.py"),
        "--model", args.model,
        "--lora_path", str(phase_output / "lora_judgment_v1"),
        "--num_samples", str(args.test_samples),
        "--num_trials", str(args.num_trials),
        "--inference_batch_size", str(args.inference_batch_size),
        "--split", "validation",  # Test generalization
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    # Record phase result
    pipeline.record_phase_result(
        phase_name="phase1_judgment",
        status="completed",
        output_paths={
            "responses": str(phase_output / "responses.jsonl"),
            "training_data": str(phase_output / "training_data.jsonl"),
            "lora_judgment_v1": str(phase_output / "lora_judgment_v1"),
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
    ]
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

    # Use merged model from Phase 2
    base_model = phase2_output / "base_with_knowledge"
    if not base_model.exists():
        print(f"Warning: Merged model not found at {base_model}")
        print(f"Using original model: {args.model}")
        base_model = args.model

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
    ]
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

    # Data params
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Training samples")
    parser.add_argument("--test_samples", type=int, default=100,
                        help="Test samples for evaluation")
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--inference_batch_size", type=int, default=16)

    # Training params
    parser.add_argument("--epochs", type=int, default=3,
                        help="Epochs for judgment training")
    parser.add_argument("--knowledge_epochs", type=int, default=5,
                        help="Epochs for knowledge training (Phase 2)")
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_base = project_root / args.output_dir

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

    # Execute phases
    for phase in phases_to_run:
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
