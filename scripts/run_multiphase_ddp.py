"""
Run Multi-phase Pipeline with DDP Support

DDP version of the three-phase training workflow. Uses DistributedDataParallel
for multi-GPU training where each GPU processes different samples.

Each phase's each step checks if already completed, loads previous results
if done (unless --force flag is provided).

Usage:
    # Run with torchrun
    torchrun --nproc_per_node=8 run_multiphase_ddp.py --model Qwen/Qwen2.5-7B-Instruct --ddp

    # Run specific phase
    torchrun --nproc_per_node=8 run_multiphase_ddp.py --phase 2 --ddp

    # Force re-run even if completed
    torchrun --nproc_per_node=8 run_multiphase_ddp.py --force --ddp
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist

from src.pipeline import MultiPhasePipeline
from src.dataset_builder import load_from_jsonl, save_to_jsonl
from src.data_loader import load_triviaqa
from src.multi_gpu_inference import create_inference
from src.evaluator import evaluate_responses, classify_ability, is_correct
from src.label_generator import build_training_dataset, SYSTEM_PROMPT
from src.knowledge_trainer import build_qa_dataset
from src.trainer import setup_model_for_training
from src.adapter_utils import merge_adapter_into_base
from tqdm import tqdm


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    """Initialize distributed training."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank
    return 0


def cleanup_ddp():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def generate_experiment_name(model: str, dataset: str, train_samples: int, test_samples: int) -> str:
    """Generate experiment name from parameters (deterministic, no timestamp)."""
    model_short = model.split("/")[-1].replace("-Instruct", "")
    dataset_short = dataset.replace("/", "_")
    return f"{model_short}_{dataset_short}_train{train_samples}_test{test_samples}"


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
        "inference_batch_size": args.inference_batch_size,
        "learning_rate": args.lr,
        "no_lora": args.no_lora,
        "adaptive": args.adaptive,
        "max_steps_per_sample": args.max_steps_per_sample,
        "ddp": args.ddp,
    }

    log_path = output_dir / "config.json"
    with open(log_path, 'w') as f:
        json.dump(config, f, indent=2)

    if is_main_process():
        print(f"Config saved to: {log_path}")
    return config


# ============== Step completion checkers ==============

def is_step_completed(phase: int, step: str, pipeline: MultiPhasePipeline, args) -> bool:
    """Check if a specific step within a phase has been completed."""
    if phase == 1:
        phase_output = pipeline.get_phase_output_dir("phase1_judgment")
        if step == "1.1_responses":
            return (phase_output / "responses.jsonl").exists()
        elif step == "1.2_training_data":
            return (phase_output / "training_data.jsonl").exists()
        elif step == "1.3_train":
            if args.no_lora:
                return (phase_output / "judgment_v1" / "config.json").exists()
            else:
                return (phase_output / "judgment_v1" / "adapter_config.json").exists()

    elif phase == 2:
        phase_output = pipeline.get_phase_output_dir("phase2_knowledge")
        if step == "2.1_qa_data":
            return (phase_output / "qa_training_data.jsonl").exists()
        elif step == "2.2_train":
            if args.no_lora:
                return (phase_output / "knowledge" / "config.json").exists()
            else:
                return (phase_output / "knowledge" / "adapter_config.json").exists()
        elif step == "2.3_merge":
            if args.no_lora:
                return (phase_output / "knowledge" / "config.json").exists()
            else:
                return (phase_output / "base_with_knowledge" / "config.json").exists()

    elif phase == 3:
        phase_output = pipeline.get_phase_output_dir("phase3_judgment")
        if step == "3.1_responses":
            return (phase_output / "responses_post_knowledge.jsonl").exists()
        elif step == "3.2_training_data":
            return (phase_output / "training_data_v2.jsonl").exists()
        elif step == "3.3_train":
            if args.no_lora:
                return (phase_output / "judgment_v2" / "config.json").exists()
            else:
                return (phase_output / "judgment_v2" / "adapter_config.json").exists()

    return False


def is_phase_completed(phase: int, pipeline: MultiPhasePipeline, args) -> bool:
    """Check if a phase has been fully completed."""
    if phase == 1:
        return is_step_completed(1, "1.3_train", pipeline, args)
    elif phase == 2:
        return is_step_completed(2, "2.3_merge", pipeline, args)
    elif phase == 3:
        return is_step_completed(3, "3.3_train", pipeline, args)
    return False


# ============== Phase 1: Initial Judgment Training ==============

def run_phase1(args, pipeline: MultiPhasePipeline, local_rank: int):
    """Run Phase 1: Initial judgment training."""
    if is_main_process():
        print("\n" + "=" * 60)
        print("Phase 1: Initial Judgment Training (DDP)")
        print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    phase_output = pipeline.get_phase_output_dir("phase1_judgment")

    # Step 1.1: Collect responses
    # NOTE: Inference runs ONLY on rank 0 to avoid multi-GPU conflicts in DDP mode
    responses_path = phase_output / "responses.jsonl"
    if not args.force and is_step_completed(1, "1.1_responses", pipeline, args):
        if is_main_process():
            print(f"\n[Step 1.1] Responses already collected, loading from {responses_path}")
        # All processes load from file after barrier
    else:
        if is_main_process():
            print(f"\n[Step 1.1] Collecting responses from train split...")

            # Load questions
            raw_samples = load_triviaqa(split="train", num_samples=args.num_samples)

            # Use multi-GPU inference for response collection (only rank 0)
            inference = create_inference(
                model_name=args.model,
                inference_batch_size=args.inference_batch_size,
                temperature=1.0,
                multi_gpu=True,
                num_gpus=args.num_gpus,
            )

            samples = inference.batch_inference(
                samples=raw_samples,
                num_trials=args.num_trials,
                prompt_formatter=lambda s: f"Question: {s['question']}\nAnswer:",
            )

            # Evaluate and classify
            for sample in samples:
                gold_answers = sample.get("normalized_answers", sample.get("answers", []))
                evaluation = evaluate_responses(sample["responses"], gold_answers)
                sample["evaluation"] = evaluation
                sample["ability"] = classify_ability(evaluation["correct_count"], evaluation["total"])

            if hasattr(inference, 'shutdown'):
                inference.shutdown()
            del inference
            torch.cuda.empty_cache()

            # Save responses
            save_to_jsonl(samples, str(responses_path))
            print(f"Saved {len(samples)} samples to {responses_path}")

    # Synchronize: wait for rank 0 to finish inference
    if dist.is_initialized():
        dist.barrier()

    # All processes load the saved responses
    samples = load_from_jsonl(str(responses_path))

    # Step 1.2: Build training data
    training_data_path = phase_output / "training_data.jsonl"
    if not args.force and is_step_completed(1, "1.2_training_data", pipeline, args):
        if is_main_process():
            print(f"\n[Step 1.2] Training data already built, loading from {training_data_path}")
        training_data = load_from_jsonl(str(training_data_path))
    else:
        if is_main_process():
            print(f"\n[Step 1.2] Building judgment training data...")
        training_data = build_training_dataset(samples)
        if is_main_process():
            save_to_jsonl(training_data, str(training_data_path))
            print(f"Saved {len(training_data)} training samples")

    if dist.is_initialized():
        dist.barrier()

    # Step 1.3: Train judgment (DDP)
    adapter_path = phase_output / "judgment_v1"
    if not args.force and is_step_completed(1, "1.3_train", pipeline, args):
        if is_main_process():
            print(f"\n[Step 1.3] Judgment model already trained at {adapter_path}")
    else:
        if is_main_process():
            print(f"\n[Step 1.3] Training judgment ability with DDP...")

        model, tokenizer = setup_model_for_training(
            args.model,
            use_lora=not args.no_lora,
            ddp=args.ddp,
            local_rank=local_rank,
        )

        if args.adaptive:
            from src.ddp_adaptive_trainer import DDPAdaptiveJudgmentTrainer

            trainer = DDPAdaptiveJudgmentTrainer(
                model=model,
                tokenizer=tokenizer,
                learning_rate=args.lr,
                local_rank=local_rank,
            )

            stats = trainer.train_dataset(
                training_data,
                system_prompt=SYSTEM_PROMPT,
                num_epochs=args.epochs,
                skip_correct=True,
            )

            if is_main_process():
                trainer.raw_model.save_pretrained(str(adapter_path))
                tokenizer.save_pretrained(str(adapter_path))
                print(f"Saved judgment model to {adapter_path}")
                print(f"Final stats: {stats['per_epoch'][-1]}")

        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Record phase result
    if is_main_process():
        pipeline.record_phase_result(
            phase_name="phase1_judgment",
            status="completed",
            metrics={},
            output_paths={
                "responses": str(responses_path),
                "training_data": str(training_data_path),
                "judgment_v1": str(adapter_path),
            }
        )
        pipeline.state.current_phase = 1
        pipeline._save_state()
        print("\nPhase 1 completed!")


# ============== Phase 2: Knowledge Learning ==============

def run_phase2(args, pipeline: MultiPhasePipeline, local_rank: int):
    """Run Phase 2: Knowledge learning."""
    if is_main_process():
        print("\n" + "=" * 60)
        print("Phase 2: Knowledge Learning (DDP)")
        print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    phase1_output = pipeline.get_phase_output_dir("phase1_judgment")
    phase2_output = pipeline.get_phase_output_dir("phase2_knowledge")

    # Load Phase 1 responses
    input_path = phase1_output / "responses.jsonl"
    if not input_path.exists():
        input_path = project_root / "data/step1_responses.jsonl"
    samples = load_from_jsonl(str(input_path))

    # Step 2.1: Build QA dataset
    qa_data_path = phase2_output / "qa_training_data.jsonl"
    if not args.force and is_step_completed(2, "2.1_qa_data", pipeline, args):
        if is_main_process():
            print(f"\n[Step 2.1] QA data already built, loading from {qa_data_path}")
        qa_data = load_from_jsonl(str(qa_data_path))
    else:
        if is_main_process():
            print(f"\n[Step 2.1] Building QA training dataset...")
        qa_data = build_qa_dataset(samples)
        if is_main_process():
            save_to_jsonl(qa_data, str(qa_data_path))
            print(f"Created {len(qa_data)} QA training samples")

    if dist.is_initialized():
        dist.barrier()

    # Step 2.2: Train knowledge model (DDP)
    adapter_path = phase2_output / "knowledge"
    if not args.force and is_step_completed(2, "2.2_train", pipeline, args):
        if is_main_process():
            print(f"\n[Step 2.2] Knowledge model already trained at {adapter_path}")
    else:
        if is_main_process():
            print(f"\n[Step 2.2] Training knowledge model with DDP...")

        model, tokenizer = setup_model_for_training(
            args.model,
            use_lora=not args.no_lora,
            ddp=args.ddp,
            local_rank=local_rank,
        )

        # Prepare adaptive samples
        adaptive_samples = []
        for sample in qa_data:
            question = sample["messages"][1]["content"]
            answer = sample["messages"][2]["content"]
            adaptive_samples.append({
                "messages": sample["messages"],
                "question": question,
                "answers": [answer],
                "normalized_answers": [answer],
                "original_ability": sample.get("original_ability", ""),
            })

        if args.adaptive:
            from src.ddp_adaptive_trainer import DDPAdaptiveKnowledgeTrainer

            trainer = DDPAdaptiveKnowledgeTrainer(
                model=model,
                tokenizer=tokenizer,
                learning_rate=args.lr,
                local_rank=local_rank,
            )

            stats = trainer.train_dataset(
                adaptive_samples,
                num_epochs=args.knowledge_epochs,
                filter_by_ability=["cannot", "uncertain"],
                skip_correct=True,
            )

            if is_main_process():
                trainer.raw_model.save_pretrained(str(adapter_path))
                tokenizer.save_pretrained(str(adapter_path))
                print(f"Saved knowledge model to {adapter_path}")
                print(f"Final stats: {stats['per_epoch'][-1]}")

        del model
        del tokenizer
        torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()

    # Step 2.3: Merge adapter (main process only)
    merged_path = phase2_output / "base_with_knowledge"
    if not args.force and is_step_completed(2, "2.3_merge", pipeline, args):
        if is_main_process():
            print(f"\n[Step 2.3] Merged model already exists at {merged_path}")
    else:
        if is_main_process():
            if args.no_lora:
                print(f"\n[Step 2.3] Full fine-tuning: model saved directly to {adapter_path}")
                merged_path = adapter_path
            else:
                print(f"\n[Step 2.3] Merging adapter into base model...")
                merge_adapter_into_base(
                    args.model,
                    str(adapter_path),
                    str(merged_path)
                )
                print(f"Merged model saved to {merged_path}")

    # Record phase result
    if is_main_process():
        pipeline.record_phase_result(
            phase_name="phase2_knowledge",
            status="completed",
            metrics={},
            output_paths={
                "qa_data": str(qa_data_path),
                "knowledge": str(adapter_path),
                "base_with_knowledge": str(merged_path) if not args.no_lora else str(adapter_path),
            }
        )
        pipeline.state.current_phase = 2
        pipeline._save_state()
        print("\nPhase 2 completed!")


# ============== Phase 3: Update Judgment ==============

def run_phase3(args, pipeline: MultiPhasePipeline, local_rank: int):
    """Run Phase 3: Update judgment with knowledge."""
    if is_main_process():
        print("\n" + "=" * 60)
        print("Phase 3: Update Judgment with Knowledge (DDP)")
        print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    phase1_output = pipeline.get_phase_output_dir("phase1_judgment")
    phase2_output = pipeline.get_phase_output_dir("phase2_knowledge")
    phase3_output = pipeline.get_phase_output_dir("phase3_judgment")

    # Get knowledge model path
    if args.no_lora:
        base_model = str(phase2_output / "knowledge")
    else:
        base_model = str(phase2_output / "base_with_knowledge")

    if not Path(base_model).exists():
        if is_main_process():
            print(f"Warning: Knowledge model not found at {base_model}, using original")
        base_model = args.model

    # Load Phase 1 responses for questions
    input_path = phase1_output / "responses.jsonl"
    if not input_path.exists():
        input_path = project_root / "data/step1_responses.jsonl"
    original_samples = load_from_jsonl(str(input_path))[:args.num_samples]

    # Step 3.1: Re-collect responses with knowledge model
    # NOTE: Inference runs ONLY on rank 0 to avoid multi-GPU conflicts in DDP mode
    responses_path = phase3_output / "responses_post_knowledge.jsonl"
    if not args.force and is_step_completed(3, "3.1_responses", pipeline, args):
        if is_main_process():
            print(f"\n[Step 3.1] Responses already collected, loading from {responses_path}")
        # All processes load from file after barrier
    else:
        if is_main_process():
            print(f"\n[Step 3.1] Re-collecting responses with knowledge model...")

            inference = create_inference(
                model_name=base_model,
                inference_batch_size=args.inference_batch_size,
                temperature=1.0,
                multi_gpu=True,
                num_gpus=args.num_gpus,
            )

            new_samples = inference.batch_inference(
                samples=original_samples,
                num_trials=args.num_trials,
                prompt_formatter=lambda s: f"Question: {s['question']}\nAnswer:",
            )

            for sample in new_samples:
                gold_answers = sample.get("normalized_answers", sample.get("answers", []))
                evaluation = evaluate_responses(sample["responses"], gold_answers)
                sample["evaluation"] = evaluation
                sample["ability"] = classify_ability(evaluation["correct_count"], evaluation["total"])

            if hasattr(inference, 'shutdown'):
                inference.shutdown()
            del inference
            torch.cuda.empty_cache()

            save_to_jsonl(new_samples, str(responses_path))

            # Show distribution change
            new_dist = {}
            for s in new_samples:
                ability = s.get("ability", "unknown")
                new_dist[ability] = new_dist.get(ability, 0) + 1
            print(f"New ability distribution: {new_dist}")

    # Synchronize: wait for rank 0 to finish inference
    if dist.is_initialized():
        dist.barrier()

    # All processes load the saved responses
    new_samples = load_from_jsonl(str(responses_path))

    # Step 3.2: Build new training data
    training_data_path = phase3_output / "training_data_v2.jsonl"
    if not args.force and is_step_completed(3, "3.2_training_data", pipeline, args):
        if is_main_process():
            print(f"\n[Step 3.2] Training data already built, loading from {training_data_path}")
        training_data = load_from_jsonl(str(training_data_path))
    else:
        if is_main_process():
            print(f"\n[Step 3.2] Building new judgment training data...")
        training_data = build_training_dataset(new_samples)
        if is_main_process():
            save_to_jsonl(training_data, str(training_data_path))
            print(f"Saved {len(training_data)} training samples")

    if dist.is_initialized():
        dist.barrier()

    # Step 3.3: Train updated judgment (DDP)
    adapter_path = phase3_output / "judgment_v2"
    if not args.force and is_step_completed(3, "3.3_train", pipeline, args):
        if is_main_process():
            print(f"\n[Step 3.3] Judgment v2 model already trained at {adapter_path}")
    else:
        if is_main_process():
            print(f"\n[Step 3.3] Training judgment v2 with DDP...")

        model, tokenizer = setup_model_for_training(
            base_model,
            use_lora=not args.no_lora,
            ddp=args.ddp,
            local_rank=local_rank,
        )

        if args.adaptive:
            from src.ddp_adaptive_trainer import DDPAdaptiveJudgmentTrainer

            trainer = DDPAdaptiveJudgmentTrainer(
                model=model,
                tokenizer=tokenizer,
                learning_rate=args.lr,
                local_rank=local_rank,
            )

            stats = trainer.train_dataset(
                training_data,
                system_prompt=SYSTEM_PROMPT,
                num_epochs=args.epochs,
                skip_correct=True,
            )

            if is_main_process():
                trainer.raw_model.save_pretrained(str(adapter_path))
                tokenizer.save_pretrained(str(adapter_path))
                print(f"Saved judgment v2 model to {adapter_path}")
                print(f"Final stats: {stats['per_epoch'][-1]}")

        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Record phase result
    if is_main_process():
        pipeline.record_phase_result(
            phase_name="phase3_judgment",
            status="completed",
            metrics={},
            output_paths={
                "responses": str(responses_path),
                "training_data": str(training_data_path),
                "judgment_v2": str(adapter_path),
            }
        )
        pipeline.state.current_phase = 3
        pipeline._save_state()
        print("\nPhase 3 completed!")


def main():
    parser = argparse.ArgumentParser(description="Run Multi-phase Pipeline with DDP")

    # Experiment
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./experiments")

    # Model and dataset
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", type=str, default="triviaqa")

    # Phase selection
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if step already completed")

    # Data params
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=100)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--inference_batch_size", type=int, default=16)

    # Training params
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--knowledge_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--adaptive", action="store_true", default=True)
    parser.add_argument("--max_steps_per_sample", type=int, default=10)

    # DDP params
    parser.add_argument("--ddp", action="store_true",
                        help="Use DDP for multi-GPU training")
    parser.add_argument("--num_gpus", type=int, default=None)

    args = parser.parse_args()

    # Setup DDP
    local_rank = 0
    if args.ddp:
        local_rank = setup_ddp()
        if is_main_process():
            print(f"DDP initialized with {dist.get_world_size()} GPUs")

    project_root = Path(__file__).resolve().parent.parent
    output_base = project_root / args.output_dir

    # Auto-generate experiment name
    if args.experiment is None:
        experiment_name = generate_experiment_name(
            model=args.model,
            dataset=args.dataset,
            train_samples=args.num_samples,
            test_samples=args.test_samples
        )
    else:
        experiment_name = args.experiment

    # Create experiment directory (main process only)
    experiment_dir = output_base / experiment_name
    if is_main_process():
        experiment_dir.mkdir(parents=True, exist_ok=True)
        save_config_log(experiment_dir, args, experiment_name)

    # Synchronize
    if dist.is_initialized():
        dist.barrier()

    # Create pipeline
    pipeline = MultiPhasePipeline(
        experiment_name=experiment_name,
        base_model=args.model,
        output_dir=str(output_base),
        config=vars(args)
    )

    if is_main_process():
        print("=" * 60)
        print(f"Multi-phase Pipeline (DDP)")
        print("=" * 60)
        print(f"Experiment: {experiment_name}")
        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Train samples: {args.num_samples}")
        print(f"Test samples: {args.test_samples}")
        print(f"Output: {pipeline.output_dir}")
        print(f"DDP: {args.ddp}")
        if args.ddp:
            print(f"World size: {dist.get_world_size()}")
        print("=" * 60)

    # Determine phases to run
    if args.phase:
        phases_to_run = [args.phase]
    elif args.resume:
        start_phase = pipeline.state.current_phase + 1
        phases_to_run = list(range(start_phase, 4))
        if not phases_to_run:
            if is_main_process():
                print("All phases already completed!")
                pipeline.print_summary()
            cleanup_ddp()
            return
    else:
        phases_to_run = [1, 2, 3]

    if is_main_process():
        print(f"Phases to run: {phases_to_run}")
        if args.force:
            print("Force mode: will re-run all steps")

    # Execute phases
    for phase in phases_to_run:
        # Check if phase completed (unless --force)
        if not args.force and is_phase_completed(phase, pipeline, args):
            if is_main_process():
                print(f"\n{'=' * 60}")
                print(f"Phase {phase} already completed, skipping...")
                print(f"(Use --force to re-run)")
                print(f"{'=' * 60}")
            continue

        if phase == 1:
            run_phase1(args, pipeline, local_rank)
        elif phase == 2:
            run_phase2(args, pipeline, local_rank)
        elif phase == 3:
            run_phase3(args, pipeline, local_rank)

        # Synchronize between phases
        if dist.is_initialized():
            dist.barrier()

    # Final summary
    if is_main_process():
        print("\n" + "=" * 60)
        print("Pipeline Summary")
        print("=" * 60)
        pipeline.print_summary()

    cleanup_ddp()


if __name__ == "__main__":
    main()
