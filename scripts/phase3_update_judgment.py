"""
Phase 3: Update Judgment with Knowledge

After the model learns knowledge in Phase 2, re-train judgment ability.
The model should now be more confident since it actually knows the answers.

Supports training modes:
- Standard: Fixed epochs with batch training
- Adaptive: Train each sample until correct
- DDP: Multi-GPU training with gradient synchronization

Steps:
3.1 Re-collect responses using base_with_knowledge model
3.2 Build new labels (most should be "yes" now)
3.3 Train updated judgment -> LoRA_judgment_v2
3.4 Final evaluation
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_triviaqa
from src.multi_gpu_inference import create_inference
from src.evaluator import evaluate_responses, classify_ability, is_correct
from src.dataset_builder import load_from_jsonl, save_to_jsonl, prepare_dataset_for_training
from src.label_generator import build_training_dataset, SYSTEM_PROMPT
from src.trainer import setup_model_for_training, train_metacognition
from src.pipeline import MultiPhasePipeline
from tqdm import tqdm
import re
import torch


def test_qa_accuracy(
    model_path: str,
    split: str = "train",
    num_samples: int = 100,
    num_trials: int = 5,
    inference_batch_size: int = 16,
    num_gpus: int = None,
) -> dict:
    """
    Test QA accuracy on a dataset split using batch inference.
    """
    print(f"  Testing QA accuracy on {split} split ({num_samples} samples)...")

    samples = load_triviaqa(split=split, num_samples=num_samples)

    # Filter samples with valid answers
    valid_samples = []
    for sample in samples:
        gold_answers = sample.get("normalized_answers", sample.get("answers", []))
        if gold_answers:
            valid_samples.append(sample)

    if not valid_samples:
        return {"qa_accuracy": 0, "qa_correct": 0, "qa_total": 0}

    print(f"  Building {len(valid_samples)} × {num_trials} = {len(valid_samples) * num_trials} prompts...")

    # Build all prompts at once
    all_prompts = []
    for sample in valid_samples:
        prompt = f"Question: {sample['question']}\nAnswer:"
        all_prompts.extend([prompt] * num_trials)

    # Create inference instance (auto multi-GPU)
    inference = create_inference(
        model_name=model_path,
        inference_batch_size=inference_batch_size,
        temperature=1.0,
        num_gpus=num_gpus,
    )

    # Batch generate all responses at once
    print(f"  Running batch inference...")
    all_responses = inference.generate_batch(all_prompts)

    # Clean up
    if hasattr(inference, 'shutdown'):
        inference.shutdown()
    del inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Evaluate results
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
        "qa_accuracy": accuracy * 100,
        "qa_correct": correct_count,
        "qa_total": total,
    }


def parse_ability_prediction(response: str) -> str:
    """Parse ability prediction from model response."""
    response_lower = response.lower()

    # Look for boxed format first
    boxed_match = re.search(r'\\boxed\{(\w+)\}', response)
    if boxed_match:
        pred = boxed_match.group(1).lower()
        if pred in ["yes", "can"]:
            return "can"
        elif pred in ["uncertain", "maybe"]:
            return "uncertain"
        elif pred in ["no", "cannot"]:
            return "cannot"

    # Fallback to keyword matching
    if "yes" in response_lower or "can answer" in response_lower:
        return "can"
    elif "uncertain" in response_lower or "not sure" in response_lower:
        return "uncertain"
    else:
        return "cannot"


def collect_responses_with_model(
    model_path: str,
    samples: list,
    num_trials: int = 5,
    inference_batch_size: int = 16,
    num_gpus: int = None
):
    """
    Collect responses using the knowledge-augmented model.
    Returns samples with responses, evaluation, and ability.
    """
    print(f"\nCollecting responses using model: {model_path}")
    print(f"Samples: {len(samples)}, Trials per sample: {num_trials}")

    inference = create_inference(
        model_name=model_path,
        inference_batch_size=inference_batch_size,
        temperature=1.0,
        num_gpus=num_gpus,
    )

    # Batch inference
    samples_with_responses = inference.batch_inference(
        samples=samples,
        num_trials=num_trials,
        prompt_formatter=lambda s: f"Question: {s['question']}\nAnswer:",
    )

    # Evaluate responses
    results = []
    for sample in samples_with_responses:
        gold_answers = sample.get("normalized_answers", sample.get("answers", []))
        evaluation = evaluate_responses(sample["responses"], gold_answers)
        ability = classify_ability(evaluation["correct_count"], evaluation["total"])

        result = sample.copy()
        result["evaluation"] = evaluation
        result["ability"] = ability
        results.append(result)

    # Clean up GPU memory
    if hasattr(inference, 'shutdown'):
        inference.shutdown()
    del inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def evaluate_judgment(
    model_path: str,
    samples: list,
    adapter_path: str = None,
    inference_batch_size: int = 16,
    num_gpus: int = None,
):
    """
    Evaluate judgment accuracy of the model using batch inference.
    """
    print(f"\nEvaluating judgment accuracy on {len(samples)} samples...")
    print(f"Model: {model_path}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")

    # Build all judgment prompts
    print(f"Building {len(samples)} judgment prompts...")
    all_prompts = []
    for sample in samples:
        question = sample["question"]
        # Use ChatML format directly
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        prompt += f"<|im_start|>user\nCan you answer this question correctly?\n\nQuestion: {question}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        all_prompts.append(prompt)

    # Create inference instance (auto multi-GPU)
    inference = create_inference(
        model_name=model_path,
        inference_batch_size=inference_batch_size,
        temperature=0.1,  # Low temperature for judgment
        num_gpus=num_gpus,
        lora_path=adapter_path,
    )

    # Batch generate all responses
    print("Running batch inference...")
    all_responses = inference.generate_batch(all_prompts)

    # Clean up
    if hasattr(inference, 'shutdown'):
        inference.shutdown()
    del inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Parse results
    results = []
    abilities = ["can", "uncertain", "cannot"]

    for i, sample in enumerate(samples):
        response = all_responses[i]
        actual_ability = sample["ability"]
        predicted = parse_ability_prediction(response)

        results.append({
            "question": sample["question"],
            "actual": actual_ability,
            "predicted": predicted,
            "response": response
        })

    # Compute metrics
    exact_match = sum(1 for r in results if r["predicted"] == r["actual"])
    exact_match_rate = exact_match / len(results) if results else 0

    # Confusion matrix
    confusion = {}
    for pred in abilities:
        for actual in abilities:
            confusion[f"{pred}_{actual}"] = sum(
                1 for r in results if r["predicted"] == pred and r["actual"] == actual
            )

    # Distribution
    pred_dist = {a: sum(1 for r in results if r["predicted"] == a) for a in abilities}
    actual_dist = {a: sum(1 for r in results if r["actual"] == a) for a in abilities}

    return {
        "exact_match_rate": exact_match_rate,
        "exact_match": exact_match,
        "total": len(results),
        "confusion": confusion,
        "predicted_distribution": pred_dist,
        "actual_distribution": actual_dist,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Update Judgment")

    project_root = Path(__file__).resolve().parent.parent

    # Model paths
    parser.add_argument("--base_model", type=str,
                        default=str(project_root / "outputs/phase2_knowledge/base_with_knowledge"),
                        help="Path to knowledge-augmented model (from Phase 2)")
    parser.add_argument("--original_base", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Original base model (for comparison)")

    # Data paths
    parser.add_argument("--input", type=str,
                        default=str(project_root / "data/step1_responses.jsonl"),
                        help="Original training data (same as Phase 1)")
    parser.add_argument("--output_dir", type=str,
                        default=str(project_root / "outputs/phase3_judgment"))

    # Training params
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="1e-4 for LoRA, 1e-5 for full fine-tuning")
    parser.add_argument("--no_lora", action="store_true",
                        help="Disable LoRA for full fine-tuning")
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--adaptive", action="store_true", default=True,
                        help="Use adaptive training (train each sample until correct)")
    parser.add_argument("--max_steps_per_sample", type=int, default=10,
                        help="Max training steps per sample in adaptive mode")
    parser.add_argument("--skip_correct", action="store_true", default=True,
                        help="Skip samples model already judges correctly (test before each epoch)")

    # Evaluation
    parser.add_argument("--test_samples", type=int, default=100)

    # GPU params
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs to use for inference (default: all available)")

    # Pipeline integration
    parser.add_argument("--experiment", type=str, default=None)

    # DDP training
    parser.add_argument("--ddp", action="store_true",
                        help="Use DDP for multi-GPU training (launch with torchrun)")

    args = parser.parse_args()

    # Setup DDP if enabled
    local_rank = 0
    if args.ddp:
        from src.ddp_adaptive_trainer import setup_ddp, is_main_process
        local_rank = setup_ddp()
        if is_main_process():
            print(f"DDP initialized with LOCAL_RANK={local_rank}")
    else:
        is_main_process = lambda: True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    pipeline = None
    if args.experiment:
        pipeline = MultiPhasePipeline(
            experiment_name=args.experiment,
            base_model=args.original_base,
            output_dir=str(project_root / "experiments")
        )
        output_dir = pipeline.get_phase_output_dir("phase3_judgment")

    # Step 3.1: Re-collect responses with knowledge model
    print("=" * 60)
    print("Phase 3: Update Judgment with Knowledge")
    print("=" * 60)

    # Load original questions
    print(f"\nLoading original data from {args.input}")
    original_samples = load_from_jsonl(args.input)[:args.num_samples]
    print(f"Using {len(original_samples)} samples")

    # Show original ability distribution
    original_dist = {}
    for s in original_samples:
        ability = s.get("ability", "unknown")
        original_dist[ability] = original_dist.get(ability, 0) + 1
    print(f"Original ability distribution: {original_dist}")

    # Re-collect with knowledge model
    print(f"\nRe-collecting responses with knowledge model...")
    new_samples = collect_responses_with_model(
        model_path=args.base_model,
        samples=original_samples,
        num_trials=args.num_trials,
        inference_batch_size=args.inference_batch_size,
        num_gpus=args.num_gpus
    )

    # Show new ability distribution
    new_dist = {}
    for s in new_samples:
        ability = s.get("ability", "unknown")
        new_dist[ability] = new_dist.get(ability, 0) + 1
    print(f"\nNew ability distribution: {new_dist}")
    print("(Model should know more now -> more 'can')")

    # Save new responses
    responses_path = output_dir / "responses_post_knowledge.jsonl"
    save_to_jsonl(new_samples, str(responses_path))

    # Step 3.2: Build new training data
    print("\nBuilding new judgment training data...")
    training_data = build_training_dataset(new_samples)
    training_data_path = output_dir / "training_data_v2.jsonl"
    save_to_jsonl(training_data, str(training_data_path))
    print(f"Saved {len(training_data)} training samples to {training_data_path}")

    # Step 3.3: Train updated judgment
    if is_main_process():
        print(f"\nSetting up model for judgment training...")
        print(f"Base: {args.base_model}")
        print(f"Training mode: {'Full fine-tuning' if args.no_lora else 'LoRA'}")
        print(f"Adaptive training: {args.adaptive}")
        print(f"DDP: {args.ddp}")
    model, tokenizer = setup_model_for_training(
        args.base_model,
        use_lora=not args.no_lora,
        ddp=args.ddp,
        local_rank=local_rank,
    )

    adapter_path = output_dir / "judgment_v2"

    if args.adaptive:
        if args.ddp:
            # DDP adaptive training
            from src.ddp_adaptive_trainer import DDPAdaptiveJudgmentTrainer, is_main_process, cleanup_ddp

            if is_main_process():
                print(f"\nUsing DDP adaptive training")
                print(f"Epochs: {args.epochs}")
                print(f"Skip already correct: {args.skip_correct}")

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
                skip_correct=args.skip_correct,
            )

            # Save model (only main process)
            if is_main_process():
                print(f"\nSaving model to {adapter_path}")
                # Get the raw model from DDP wrapper
                raw_model = trainer.raw_model
                raw_model.save_pretrained(str(adapter_path))
                tokenizer.save_pretrained(str(adapter_path))
                print(f"\nDDP adaptive training complete!")
                print(f"Final stats: {stats['per_epoch'][-1]}")

        else:
            # Single-GPU adaptive training
            from src.adaptive_trainer import AdaptiveJudgmentTrainer

            print(f"\nUsing adaptive training (max {args.max_steps_per_sample} steps per sample)")
            print(f"Epochs: {args.epochs}")
            print(f"Skip already correct: {args.skip_correct}")

            trainer = AdaptiveJudgmentTrainer(
                model=model,
                tokenizer=tokenizer,
                learning_rate=args.lr,
                max_steps_per_sample=args.max_steps_per_sample,
            )

            print(f"\nTraining on {len(training_data)} samples...")
            stats = trainer.train_dataset(
                training_data,
                system_prompt=SYSTEM_PROMPT,
                num_epochs=args.epochs,
                skip_correct=args.skip_correct,
            )

            # Save model
            print(f"\nSaving model to {adapter_path}")
            model.save_pretrained(str(adapter_path))
            tokenizer.save_pretrained(str(adapter_path))

            print(f"\nAdaptive training complete!")
            print(f"Final stats: {stats['per_epoch'][-1]}")

    else:
        # Standard training
        print("Preparing dataset...")
        datasets = prepare_dataset_for_training(training_data, tokenizer)
        print(f"Train: {len(datasets['train'])}, Val: {len(datasets['validation'])}")

        print(f"\nTraining judgment v2...")
        train_metacognition(
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            val_dataset=datasets["validation"],
            output_dir=str(adapter_path),
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_lora=not args.no_lora,
        )

    # DDP cleanup and post-training steps (only on main process for DDP)
    if args.ddp:
        from src.ddp_adaptive_trainer import cleanup_ddp
        # Only main process continues with evaluation
        if not is_main_process():
            cleanup_ddp()
            return

    if is_main_process():
        print(f"Judgment model saved to {adapter_path}")

    # Clean up training model before evaluation
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if is_main_process():
        print("Cleaned up training model from GPU memory")

    # Step 3.4: Final evaluation on both train and validation splits
    if is_main_process():
        print("\n" + "=" * 60)
        print("Final Evaluation")
        print("=" * 60)

    # ===== Evaluate on TRAIN split =====
    print("\n--- Evaluation on TRAIN split (verify learning) ---")
    # Use all training samples for train evaluation (not test_samples)
    train_test_samples = new_samples  # Evaluate on all trained samples
    train_dist = {}
    for s in train_test_samples:
        ability = s.get("ability", "unknown")
        train_dist[ability] = train_dist.get(ability, 0) + 1
    print(f"Train test ability distribution: {train_dist}")

    print("\nBefore judgment v2 training (knowledge model only) on TRAIN:")
    before_train_eval = evaluate_judgment(
        model_path=args.base_model,
        samples=train_test_samples,
        inference_batch_size=args.inference_batch_size,
        num_gpus=args.num_gpus,
    )
    print(f"Exact Match: {before_train_eval['exact_match_rate']*100:.1f}%")

    print("\nAfter judgment v2 training on TRAIN:")
    if args.no_lora:
        # Full fine-tuning: model is saved directly in adapter_path
        after_train_eval = evaluate_judgment(
            model_path=str(adapter_path),
            samples=train_test_samples,
            adapter_path=None,
            inference_batch_size=args.inference_batch_size,
            num_gpus=args.num_gpus,
        )
    else:
        # LoRA: use base model + adapter
        after_train_eval = evaluate_judgment(
            model_path=args.base_model,
            samples=train_test_samples,
            adapter_path=str(adapter_path),
            inference_batch_size=args.inference_batch_size,
            num_gpus=args.num_gpus,
        )
    print(f"Exact Match: {after_train_eval['exact_match_rate']*100:.1f}%")
    print(f"Predicted: {after_train_eval['predicted_distribution']}")
    print(f"Actual: {after_train_eval['actual_distribution']}")

    train_improvement = after_train_eval['exact_match_rate'] - before_train_eval['exact_match_rate']
    print(f"Train Improvement: {train_improvement*100:+.1f}%")

    # ===== Evaluate on VALIDATION split =====
    print("\n--- Evaluation on VALIDATION split (test generalization) ---")
    val_samples = load_triviaqa(split="validation", num_samples=args.test_samples)
    print(f"Loaded {len(val_samples)} test samples from validation split")

    print(f"\nCollecting responses with knowledge model to determine actual ability...")
    val_test_samples = collect_responses_with_model(
        model_path=args.base_model,
        samples=val_samples,
        num_trials=args.num_trials,
        inference_batch_size=args.inference_batch_size,
        num_gpus=args.num_gpus,
    )

    val_dist = {}
    for s in val_test_samples:
        ability = s.get("ability", "unknown")
        val_dist[ability] = val_dist.get(ability, 0) + 1
    print(f"Validation test ability distribution: {val_dist}")

    print("\nBefore judgment v2 training (knowledge model only) on VALIDATION:")
    before_val_eval = evaluate_judgment(
        model_path=args.base_model,
        samples=val_test_samples,
        inference_batch_size=args.inference_batch_size,
        num_gpus=args.num_gpus,
    )
    print(f"Exact Match: {before_val_eval['exact_match_rate']*100:.1f}%")

    print("\nAfter judgment v2 training on VALIDATION:")
    if args.no_lora:
        # Full fine-tuning: model is saved directly in adapter_path
        after_val_eval = evaluate_judgment(
            model_path=str(adapter_path),
            samples=val_test_samples,
            adapter_path=None,
            inference_batch_size=args.inference_batch_size,
            num_gpus=args.num_gpus,
        )
    else:
        # LoRA: use base model + adapter
        after_val_eval = evaluate_judgment(
            model_path=args.base_model,
            samples=val_test_samples,
            adapter_path=str(adapter_path),
            inference_batch_size=args.inference_batch_size,
            num_gpus=args.num_gpus,
        )
    print(f"Exact Match: {after_val_eval['exact_match_rate']*100:.1f}%")
    print(f"Predicted: {after_val_eval['predicted_distribution']}")
    print(f"Actual: {after_val_eval['actual_distribution']}")

    val_improvement = after_val_eval['exact_match_rate'] - before_val_eval['exact_match_rate']
    print(f"Validation Improvement: {val_improvement*100:+.1f}%")

    # Print confusion matrix for validation
    print(f"\nConfusion Matrix (Validation):")
    c = after_val_eval['confusion']
    print(f"                      actual_can  actual_uncertain  actual_cannot")
    print(f"  predicted_can          {c['can_can']:5d}           {c['can_uncertain']:5d}            {c['can_cannot']:5d}")
    print(f"  predicted_uncertain    {c['uncertain_can']:5d}           {c['uncertain_uncertain']:5d}            {c['uncertain_cannot']:5d}")
    print(f"  predicted_cannot       {c['cannot_can']:5d}           {c['cannot_uncertain']:5d}            {c['cannot_cannot']:5d}")

    # ===== QA Evaluation =====
    print("\n" + "=" * 60)
    print("QA Accuracy Evaluation (verify knowledge preserved)")
    print("=" * 60)

    # Before judgment v2 training (knowledge model only)
    print("\nBefore judgment v2 training (knowledge model) QA:")
    qa_before_train = test_qa_accuracy(
        args.base_model, split="train", num_samples=args.test_samples,
        num_trials=args.num_trials, inference_batch_size=args.inference_batch_size,
        num_gpus=args.num_gpus
    )
    print(f"  Train QA: {qa_before_train['qa_accuracy']:.1f}%")

    qa_before_val = test_qa_accuracy(
        args.base_model, split="validation", num_samples=args.test_samples,
        num_trials=args.num_trials, inference_batch_size=args.inference_batch_size,
        num_gpus=args.num_gpus
    )
    print(f"  Validation QA: {qa_before_val['qa_accuracy']:.1f}%")

    # After judgment v2 training
    print("\nAfter judgment v2 training QA:")
    if args.no_lora:
        qa_test_model = str(adapter_path)
    else:
        # For LoRA judgment training on knowledge model, QA should remain same
        qa_test_model = args.base_model

    qa_after_train = test_qa_accuracy(
        qa_test_model, split="train", num_samples=args.test_samples,
        num_trials=args.num_trials, inference_batch_size=args.inference_batch_size,
        num_gpus=args.num_gpus
    )
    print(f"  Train QA: {qa_after_train['qa_accuracy']:.1f}%")

    qa_after_val = test_qa_accuracy(
        qa_test_model, split="validation", num_samples=args.test_samples,
        num_trials=args.num_trials, inference_batch_size=args.inference_batch_size,
        num_gpus=args.num_gpus
    )
    print(f"  Validation QA: {qa_after_val['qa_accuracy']:.1f}%")

    qa_change_train = qa_after_train['qa_accuracy'] - qa_before_train['qa_accuracy']
    qa_change_val = qa_after_val['qa_accuracy'] - qa_before_val['qa_accuracy']
    print(f"\nQA Change (should be ~0): Train {qa_change_train:+.1f}%, Val {qa_change_val:+.1f}%")

    # Record to pipeline
    if pipeline:
        pipeline.record_phase_result(
            phase_name="phase3_judgment",
            status="completed",
            metrics={
                "original_distribution": original_dist,
                "new_distribution": new_dist,
                "train_before_exact_match": before_train_eval['exact_match_rate'],
                "train_after_exact_match": after_train_eval['exact_match_rate'],
                "train_improvement": train_improvement,
                "val_before_exact_match": before_val_eval['exact_match_rate'],
                "val_after_exact_match": after_val_eval['exact_match_rate'],
                "val_improvement": val_improvement,
                "confusion_matrix": after_val_eval['confusion'],
                # QA metrics
                "qa_train_before": qa_before_train['qa_accuracy'],
                "qa_train_after": qa_after_train['qa_accuracy'],
                "qa_val_before": qa_before_val['qa_accuracy'],
                "qa_val_after": qa_after_val['qa_accuracy'],
            },
            output_paths={
                "responses": str(responses_path),
                "training_data": str(training_data_path),
                "judgment_v2": str(adapter_path),
            }
        )
        pipeline.print_summary()

    if is_main_process():
        print("\n" + "=" * 60)
        print("Phase 3 (Update Judgment) completed!")
        print("=" * 60)
        print(f"\nFinal model: {args.base_model} + {adapter_path}")

    # Final DDP cleanup
    if args.ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
