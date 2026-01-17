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
from tqdm import tqdm
import re
import subprocess
import gc
import time


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


# ============== Evaluation Functions ==============

def test_qa_accuracy(
    model_path: str,
    samples: list,
    num_trials: int = 5,
    inference_batch_size: int = 16,
    num_gpus: int = None,
    model=None,
    tokenizer=None,
) -> dict:
    """
    Test QA accuracy on given samples using batch inference.

    Args:
        model_path: Path to model (used if model/tokenizer not provided)
        samples: List of samples to test
        num_trials: Number of generation trials per sample
        inference_batch_size: Batch size for inference
        num_gpus: Number of GPUs (for multi-GPU inference when loading from path)
        model: Pre-loaded model (optional, avoids reloading)
        tokenizer: Pre-loaded tokenizer (optional, avoids reloading)

    Returns accuracy, correct count, and total count.
    """
    # Filter samples with valid answers
    valid_samples = []
    for sample in samples:
        gold_answers = sample.get("normalized_answers", sample.get("answers", []))
        if gold_answers:
            valid_samples.append(sample)

    if not valid_samples:
        return {"qa_accuracy": 0, "qa_correct": 0, "qa_total": 0}

    # Build all prompts at once: samples × num_trials
    all_prompts = []
    for sample in valid_samples:
        prompt = f"Question: {sample['question']}\nAnswer:"
        all_prompts.extend([prompt] * num_trials)

    # Use provided model or create inference instance
    if model is not None and tokenizer is not None:
        # Use provided model directly (single GPU, already loaded)
        print("Using pre-loaded model for QA accuracy test...")
        model.eval()
        all_responses = []

        # Process in batches with LEFT padding for generation
        tokenizer.padding_side = "left"
        for i in range(0, len(all_prompts), inference_batch_size):
            batch_prompts = all_prompts[i:i + inference_batch_size]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode responses
            for output in outputs:
                response = tokenizer.decode(
                    output[inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                all_responses.append(response)

        # No cleanup - caller manages the model
    else:
        # Create inference instance (auto multi-GPU)
        inference = create_inference(
            model_name=model_path,
            inference_batch_size=inference_batch_size,
            temperature=1.0,
            num_gpus=num_gpus,
        )

        # Batch generate all responses at once
        all_responses = inference.generate_batch(all_prompts)

        # Clean up inference
        if hasattr(inference, 'shutdown'):
            inference.shutdown()
        del inference
        torch.cuda.empty_cache()

    # Evaluate results: group responses back to samples
    correct_count = 0
    for i, sample in enumerate(valid_samples):
        start_idx = i * num_trials
        end_idx = start_idx + num_trials
        responses = all_responses[start_idx:end_idx]

        gold_answers = sample.get("normalized_answers", sample.get("answers", []))
        any_correct = any(is_correct(r, gold_answers) for r in responses)
        if any_correct:
            correct_count += 1

    total = len(valid_samples)
    accuracy = correct_count / total if total > 0 else 0

    return {
        "qa_accuracy": accuracy * 100,  # As percentage
        "qa_correct": correct_count,
        "qa_total": total,
    }


def run_judgment_evaluation(
    model_path: str,
    lora_path: str,
    split: str,
    num_samples: int,
    num_trials: int,
    inference_batch_size: int,
    num_gpus: int = None,
) -> dict:
    """
    Run judgment evaluation using step4_evaluate.py script.
    Returns parsed metrics from the evaluation output.
    """
    project_root = Path(__file__).resolve().parent.parent

    cmd = [
        sys.executable, str(project_root / "scripts/step4_evaluate.py"),
        "--model", model_path,
        "--lora_path", lora_path,
        "--num_samples", str(num_samples),
        "--num_trials", str(num_trials),
        "--inference_batch_size", str(inference_batch_size),
        "--split", split,
    ]
    if num_gpus is not None:
        cmd.extend(["--num_gpus", str(num_gpus)])

    # Run and capture output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    for line in process.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)

    process.wait()
    output = ''.join(output_lines)

    # Parse metrics from output
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

    return metrics


def parse_judgment_response(response: str) -> str:
    """Parse judgment response to extract ability prediction."""
    response = response.strip().lower()

    # Parse \boxed{} format
    match = re.search(r'\\boxed\{(\w+)\}', response)
    if match:
        answer = match.group(1).lower()
        if answer == "yes":
            return "can"
        elif answer == "uncertain":
            return "uncertain"
        else:
            return "cannot"

    # Fallback: check keywords
    if "yes" in response:
        return "can"
    elif "uncertain" in response:
        return "uncertain"
    else:
        return "cannot"


def evaluate_judgment_with_model(
    model,
    tokenizer,
    samples: list,
    num_trials: int = 5,
    inference_batch_size: int = 16,
) -> dict:
    """
    Evaluate judgment accuracy using a pre-loaded model.

    Args:
        model: Pre-loaded model
        tokenizer: Pre-loaded tokenizer
        samples: List of samples to evaluate
        num_trials: Number of trials per question for actual ability

    Returns:
        Dict with evaluation metrics
    """
    print(f"Evaluating judgment on {len(samples)} samples using pre-loaded model...")

    model.eval()
    tokenizer.padding_side = "left"

    # Step 1: Predict judgment abilities
    print("Step 1: Predicting judgment abilities...")
    judgment_prompts = []
    for sample in samples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Can you answer this question correctly?\n\nQuestion: {sample['question']}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        judgment_prompts.append(prompt)

    # Generate judgment predictions in batches
    predicted_abilities = []
    for i in range(0, len(judgment_prompts), inference_batch_size):
        batch_prompts = judgment_prompts[i:i + inference_batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        for output in outputs:
            response = tokenizer.decode(
                output[inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            predicted_abilities.append(parse_judgment_response(response))

    # Step 2: Generate QA responses to determine actual abilities
    print("Step 2: Generating QA responses...")
    qa_prompts = []
    for sample in samples:
        messages = [
            {"role": "system", "content": "Answer the question concisely and directly."},
            {"role": "user", "content": sample['question']}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        qa_prompts.extend([prompt] * num_trials)

    # Generate QA responses in batches
    all_qa_responses = []
    for i in range(0, len(qa_prompts), inference_batch_size):
        batch_prompts = qa_prompts[i:i + inference_batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        for output in outputs:
            response = tokenizer.decode(
                output[inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            all_qa_responses.append(response)

    # Step 3: Evaluate results
    print("Step 3: Evaluating results...")
    correct_count = 0
    pred_dist = {"can": 0, "uncertain": 0, "cannot": 0}
    actual_dist = {"can": 0, "uncertain": 0, "cannot": 0}

    for i, sample in enumerate(samples):
        # Get QA responses for this sample
        start_idx = i * num_trials
        end_idx = start_idx + num_trials
        responses = all_qa_responses[start_idx:end_idx]

        # Determine actual ability
        gold_answers = sample.get("normalized_answers", sample.get("answers", []))
        correct_responses = sum(1 for r in responses if is_correct(r, gold_answers))
        actual_ability = classify_ability(correct_responses, num_trials)

        # Get predicted ability
        predicted_ability = predicted_abilities[i]

        # Update counts
        pred_dist[predicted_ability] += 1
        actual_dist[actual_ability] += 1

        if predicted_ability == actual_ability:
            correct_count += 1

    total = len(samples)
    exact_match_rate = correct_count / total * 100 if total > 0 else 0

    print(f"\nResults:")
    print(f"  Exact match rate: {exact_match_rate:.1f}%")
    print(f"  Predicted distribution: can={pred_dist['can']}, uncertain={pred_dist['uncertain']}, cannot={pred_dist['cannot']}")
    print(f"  Actual distribution: can={actual_dist['can']}, uncertain={actual_dist['uncertain']}, cannot={actual_dist['cannot']}")

    return {
        "exact_match_rate": exact_match_rate,
        "pred_can": pred_dist["can"],
        "pred_uncertain": pred_dist["uncertain"],
        "pred_cannot": pred_dist["cannot"],
        "actual_can": actual_dist["can"],
        "actual_uncertain": actual_dist["uncertain"],
        "actual_cannot": actual_dist["cannot"],
    }


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
        "training_mode": "full_fine_tuning",  # Always full fine-tuning, no LoRA
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
            return (phase_output / "judgment_v1" / "config.json").exists()

    elif phase == 2:
        phase_output = pipeline.get_phase_output_dir("phase2_knowledge")
        if step == "2.1_qa_data":
            return (phase_output / "qa_training_data.jsonl").exists()
        elif step == "2.2_train":
            return (phase_output / "knowledge" / "config.json").exists()

    elif phase == 3:
        phase_output = pipeline.get_phase_output_dir("phase3_judgment")
        if step == "3.1_responses":
            return (phase_output / "responses_post_knowledge.jsonl").exists()
        elif step == "3.2_training_data":
            return (phase_output / "training_data_v2.jsonl").exists()
        elif step == "3.3_train":
            return (phase_output / "judgment_v2" / "config.json").exists()

    return False


def is_phase_completed(phase: int, pipeline: MultiPhasePipeline, args) -> bool:
    """Check if a phase has been fully completed."""
    if phase == 1:
        return is_step_completed(1, "1.3_train", pipeline, args)
    elif phase == 2:
        return is_step_completed(2, "2.2_train", pipeline, args)
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

    # Step 1.3: Baseline evaluation BEFORE training (GPU is free now)
    eval_results = {}
    if is_main_process():
        print("\n" + "=" * 60)
        print("Phase 1 Baseline Evaluation (Before Training)")
        print("=" * 60)

        # Baseline evaluation uses subprocess (separate process, no memory conflict)
        print("\n[Step 1.3a] Baseline JUDGMENT evaluation (train split)...")
        eval_results['before_train'] = run_judgment_evaluation(
            args.model, "none", "train", args.num_samples,
            args.num_trials, args.inference_batch_size, args.num_gpus
        )

        print("\n[Step 1.3b] Baseline JUDGMENT evaluation (validation split)...")
        eval_results['before_val'] = run_judgment_evaluation(
            args.model, "none", "validation", args.test_samples,
            args.num_trials, args.inference_batch_size, args.num_gpus
        )

        # Baseline QA
        print("\n[Step 1.3c] Baseline QA accuracy...")
        val_samples = load_triviaqa(split="validation", num_samples=args.test_samples)
        qa_before = test_qa_accuracy(
            args.model, val_samples, args.num_trials,
            args.inference_batch_size, args.num_gpus
        )
        eval_results['before_val'].update(qa_before)
        print(f"  Baseline QA: {qa_before['qa_accuracy']:.1f}%")

    if dist.is_initialized():
        dist.barrier()

    # Step 1.4: Train judgment (DDP)
    adapter_path = phase_output / "judgment_v1"
    model = None
    tokenizer = None
    raw_model = None

    if not args.force and is_step_completed(1, "1.3_train", pipeline, args):
        if is_main_process():
            print(f"\n[Step 1.4] Judgment model already trained at {adapter_path}")
            # Load trained model for evaluation
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("Loading trained model for evaluation...")
            raw_model = AutoModelForCausalLM.from_pretrained(
                str(adapter_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
    else:
        if is_main_process():
            print(f"\n[Step 1.4] Training judgment ability with DDP (full fine-tuning)...")

        model, tokenizer = setup_model_for_training(
            args.model,
            use_lora=False,  # Always use full fine-tuning
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

            # Keep raw_model for evaluation
            raw_model = trainer.raw_model

            if is_main_process():
                raw_model.save_pretrained(str(adapter_path))
                tokenizer.save_pretrained(str(adapter_path))
                print(f"Saved judgment model to {adapter_path}")
                print(f"Final stats: {stats['per_epoch'][-1]}")

    if dist.is_initialized():
        dist.barrier()

    # Step 1.5: Evaluate AFTER training using pre-loaded model
    if is_main_process() and raw_model is not None:
        print("\n" + "=" * 60)
        print("Phase 1 After-Training Evaluation (Using Pre-loaded Model)")
        print("=" * 60)

        # Load test samples
        train_test_samples = samples[:args.num_samples]  # Use training samples
        val_test_samples = load_triviaqa(split="validation", num_samples=args.test_samples)

        # After training evaluation using pre-loaded model
        print("\n[Step 1.5a] After training JUDGMENT evaluation (train split)...")
        eval_results['after_train'] = evaluate_judgment_with_model(
            raw_model, tokenizer, train_test_samples,
            args.num_trials, args.inference_batch_size
        )

        print("\n[Step 1.5b] After training JUDGMENT evaluation (validation split)...")
        eval_results['after_val'] = evaluate_judgment_with_model(
            raw_model, tokenizer, val_test_samples,
            args.num_trials, args.inference_batch_size
        )

        # Print summary
        print("\n" + "=" * 60)
        print("Phase 1 Evaluation Summary")
        print("=" * 60)
        before_train_acc = eval_results.get('before_train', {}).get('exact_match_rate', 0)
        before_val_acc = eval_results.get('before_val', {}).get('exact_match_rate', 0)
        after_train_acc = eval_results.get('after_train', {}).get('exact_match_rate', 0)
        after_val_acc = eval_results.get('after_val', {}).get('exact_match_rate', 0)
        print(f"  JUDGMENT Accuracy:")
        print(f"    Before: Train={before_train_acc:.1f}%, Val={before_val_acc:.1f}%")
        print(f"    After:  Train={after_train_acc:.1f}%, Val={after_val_acc:.1f}%")
        print(f"    Improvement: Train={after_train_acc - before_train_acc:+.1f}%, Val={after_val_acc - before_val_acc:+.1f}%")

    # Clean up model after evaluation
    if model is not None:
        del model
    if raw_model is not None:
        del raw_model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time.sleep(1.0)
    if is_main_process():
        print("Cleaned up Phase 1 models from GPU memory")

    if dist.is_initialized():
        dist.barrier()

    # Record phase result (full fine-tuning: model saved directly to judgment_v1)
    if is_main_process():
        pipeline.record_phase_result(
            phase_name="phase1_judgment",
            status="completed",
            metrics=eval_results,
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
    """Run Phase 2: Knowledge learning on top of Phase 1's judgment model."""
    if is_main_process():
        print("\n" + "=" * 60)
        print("Phase 2: Knowledge Learning (DDP)")
        print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    phase1_output = pipeline.get_phase_output_dir("phase1_judgment")
    phase2_output = pipeline.get_phase_output_dir("phase2_knowledge")

    # Determine base model for Phase 2: Use Phase 1's trained model (full fine-tuning)
    phase1_model_path = phase1_output / "judgment_v1"
    if phase1_model_path.exists():
        phase2_base_model = str(phase1_model_path)
        if is_main_process():
            print(f"Using Phase 1 trained model: {phase2_base_model}")
    else:
        # Fallback to original base model if Phase 1 output not found
        phase2_base_model = args.model
        if is_main_process():
            print(f"Warning: Phase 1 model not found at {phase1_model_path}")
            print(f"Falling back to original base model: {phase2_base_model}")

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

    # Step 2.2: Baseline evaluation BEFORE training (GPU is free now)
    eval_results = {}
    if is_main_process():
        print("\n" + "=" * 60)
        print("Phase 2 Baseline Evaluation (Before Training)")
        print("=" * 60)

        train_test_samples = samples[:args.test_samples]
        val_test_samples = load_triviaqa(split="validation", num_samples=args.test_samples)

        print("\n[Step 2.2a] Baseline QA accuracy (train split)...")
        before_train = test_qa_accuracy(
            phase2_base_model, train_test_samples, args.num_trials,
            args.inference_batch_size, args.num_gpus
        )
        print(f"  Accuracy: {before_train['qa_accuracy']:.1f}%")

        print("\n[Step 2.2b] Baseline QA accuracy (validation split)...")
        before_val = test_qa_accuracy(
            phase2_base_model, val_test_samples, args.num_trials,
            args.inference_batch_size, args.num_gpus
        )
        print(f"  Accuracy: {before_val['qa_accuracy']:.1f}%")

        eval_results['train_before_accuracy'] = before_train['qa_accuracy']
        eval_results['val_before_accuracy'] = before_val['qa_accuracy']

    if dist.is_initialized():
        dist.barrier()

    # Step 2.3: Train knowledge model (DDP)
    adapter_path = phase2_output / "knowledge"
    model = None
    tokenizer = None
    raw_model = None

    if not args.force and is_step_completed(2, "2.2_train", pipeline, args):
        if is_main_process():
            print(f"\n[Step 2.3] Knowledge model already trained at {adapter_path}")
            # Load trained model for evaluation
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("Loading trained model for evaluation...")
            raw_model = AutoModelForCausalLM.from_pretrained(
                str(adapter_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
    else:
        if is_main_process():
            print(f"\n[Step 2.3] Training knowledge model with DDP (full fine-tuning)...")
            print(f"  Base model: {phase2_base_model}")

        model, tokenizer = setup_model_for_training(
            phase2_base_model,
            use_lora=False,  # Always use full fine-tuning
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

            # Keep raw_model for evaluation
            raw_model = trainer.raw_model

            if is_main_process():
                raw_model.save_pretrained(str(adapter_path))
                tokenizer.save_pretrained(str(adapter_path))
                print(f"Saved knowledge model to {adapter_path}")
                print(f"Final stats: {stats['per_epoch'][-1]}")

    if dist.is_initialized():
        dist.barrier()

    # Step 2.4: Test knowledge acquisition using pre-loaded model (full fine-tuning, no merge needed)
    if is_main_process() and raw_model is not None:
        print("\n" + "=" * 60)
        print("Phase 2 After-Training Evaluation (Using Pre-loaded Model)")
        print("=" * 60)

        train_test_samples = samples[:args.test_samples]
        val_test_samples = load_triviaqa(split="validation", num_samples=args.test_samples)

        # Test on TRAIN split
        print("\n[Step 2.4a] After training QA accuracy (train split)...")
        after_train = test_qa_accuracy(
            str(adapter_path),
            train_test_samples, args.num_trials,
            args.inference_batch_size, args.num_gpus,
            model=raw_model, tokenizer=tokenizer
        )
        print(f"  Accuracy: {after_train['qa_accuracy']:.1f}%")
        train_improvement = after_train['qa_accuracy'] - eval_results.get('train_before_accuracy', 0)
        print(f"  Improvement: {train_improvement:+.1f}%")

        # Test on VALIDATION split
        print("\n[Step 2.4b] After training QA accuracy (validation split)...")
        after_val = test_qa_accuracy(
            str(adapter_path),
            val_test_samples, args.num_trials,
            args.inference_batch_size, args.num_gpus,
            model=raw_model, tokenizer=tokenizer
        )
        print(f"  Accuracy: {after_val['qa_accuracy']:.1f}%")
        val_improvement = after_val['qa_accuracy'] - eval_results.get('val_before_accuracy', 0)
        print(f"  Improvement: {val_improvement:+.1f}%")

        # Summary
        print("\n" + "=" * 60)
        print("Phase 2 Evaluation Summary")
        print("=" * 60)
        print(f"  QA Accuracy:")
        print(f"    Before: Train={eval_results.get('train_before_accuracy', 0):.1f}%, Val={eval_results.get('val_before_accuracy', 0):.1f}%")
        print(f"    After:  Train={after_train['qa_accuracy']:.1f}%, Val={after_val['qa_accuracy']:.1f}%")
        print(f"    Improvement: Train={train_improvement:+.1f}%, Val={val_improvement:+.1f}%")

        eval_results.update({
            "train_after_accuracy": after_train['qa_accuracy'],
            "train_improvement": train_improvement,
            "val_after_accuracy": after_val['qa_accuracy'],
            "val_improvement": val_improvement,
        })

    # Clean up model after evaluation
    if model is not None:
        del model
    if raw_model is not None:
        del raw_model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time.sleep(1.0)
    if is_main_process():
        print("Cleaned up Phase 2 models from GPU memory")

    # Record phase result (full fine-tuning: model saved directly to knowledge/)
    if is_main_process():
        pipeline.record_phase_result(
            phase_name="phase2_knowledge",
            status="completed",
            metrics=eval_results,
            output_paths={
                "qa_data": str(qa_data_path),
                "knowledge": str(adapter_path),
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

    # Get knowledge model path (full fine-tuning: model saved directly to knowledge/)
    base_model = str(phase2_output / "knowledge")

    if not Path(base_model).exists():
        if is_main_process():
            print(f"Warning: Knowledge model not found at {base_model}, using original")
        base_model = args.model
    else:
        if is_main_process():
            print(f"Using Phase 2 knowledge model: {base_model}")

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
            print(f"\n[Step 3.3] Training judgment v2 with DDP (full fine-tuning)...")
            print(f"  Base model: {base_model}")

        model, tokenizer = setup_model_for_training(
            base_model,
            use_lora=False,  # Always use full fine-tuning
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

    if dist.is_initialized():
        dist.barrier()

    # Step 3.4: Final Evaluation (only on rank 0)
    eval_results = {}
    if is_main_process():
        print("\n" + "=" * 60)
        print("Phase 3 Evaluation: Final Assessment")
        print("=" * 60)

        # Setup evaluation paths (full fine-tuning: use judgment_v2 directly)
        eval_model = str(adapter_path)
        eval_lora = "none"

        # Judgment evaluation
        print("\n[Step 3.4a] Final JUDGMENT evaluation (train split)...")
        eval_results['judgment_train'] = run_judgment_evaluation(
            eval_model, eval_lora, "train", args.num_samples,
            args.num_trials, args.inference_batch_size, args.num_gpus
        )

        print("\n[Step 3.4b] Final JUDGMENT evaluation (validation split)...")
        eval_results['judgment_val'] = run_judgment_evaluation(
            eval_model, eval_lora, "validation", args.test_samples,
            args.num_trials, args.inference_batch_size, args.num_gpus
        )

        # QA evaluation (test if knowledge was preserved)
        print("\n[Step 3.4c] Final QA accuracy (train split)...")
        train_samples_for_qa = original_samples[:args.test_samples]
        qa_train = test_qa_accuracy(
            eval_model, train_samples_for_qa,
            args.num_trials, args.inference_batch_size, args.num_gpus
        )
        eval_results['qa_train'] = qa_train
        print(f"  QA accuracy (train): {qa_train['qa_accuracy']:.1f}%")

        print("\n[Step 3.4d] Final QA accuracy (validation split)...")
        val_samples_for_qa = load_triviaqa(split="validation", num_samples=args.test_samples)
        qa_val = test_qa_accuracy(
            eval_model, val_samples_for_qa,
            args.num_trials, args.inference_batch_size, args.num_gpus
        )
        eval_results['qa_val'] = qa_val
        print(f"  QA accuracy (val): {qa_val['qa_accuracy']:.1f}%")

        # Compare with Phase 1 (before knowledge learning)
        print("\n" + "=" * 60)
        print("Phase 3 Evaluation Summary")
        print("=" * 60)
        judgment_train_acc = eval_results.get('judgment_train', {}).get('exact_match_rate', 0)
        judgment_val_acc = eval_results.get('judgment_val', {}).get('exact_match_rate', 0)
        print(f"  Final JUDGMENT Accuracy:")
        print(f"    Train: {judgment_train_acc:.1f}%")
        print(f"    Val:   {judgment_val_acc:.1f}%")
        print(f"  Final QA Accuracy:")
        print(f"    Train: {qa_train['qa_accuracy']:.1f}%")
        print(f"    Val:   {qa_val['qa_accuracy']:.1f}%")

        # Show ability distribution change
        print("\n  Ability Distribution (Post-Knowledge):")
        ability_dist = {}
        for s in new_samples:
            ability = s.get("ability", "unknown")
            ability_dist[ability] = ability_dist.get(ability, 0) + 1
        for ability, count in sorted(ability_dist.items()):
            pct = count / len(new_samples) * 100
            print(f"    {ability}: {count} ({pct:.1f}%)")

    # Record phase result
    if is_main_process():
        pipeline.record_phase_result(
            phase_name="phase3_judgment",
            status="completed",
            metrics=eval_results,
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

    # Training params (always full fine-tuning, no LoRA)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--knowledge_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for full fine-tuning")
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
