"""
Phase 2: Knowledge Learning

Train the model to learn factual knowledge by teaching it to answer questions correctly.
This is different from judgment training - here we teach knowledge, not metacognition.

Supports two training modes:
- Standard: Fixed epochs with batch training
- Adaptive: Train each sample until learned (tested after each step)
- DDP: Multi-GPU training with gradient synchronization

Steps:
2.1 Build QA training data from training samples
2.2 Train model with QA data -> LoRA_knowledge
2.3 Merge adapter into base -> base_with_knowledge
2.4 Test if model learned the knowledge
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_builder import load_from_jsonl, save_to_jsonl, prepare_dataset_for_training
from src.knowledge_trainer import build_qa_dataset
from src.trainer import setup_model_for_training, train_metacognition
from src.adapter_utils import merge_adapter_into_base
from src.multi_gpu_inference import create_inference
from src.evaluator import is_correct
from src.pipeline import MultiPhasePipeline
from src.data_loader import load_triviaqa
from tqdm import tqdm
import torch
import gc
import time


def test_knowledge_acquisition(
    model_path: str,
    samples: list,
    num_trials: int = 5,
    inference_batch_size: int = 16,
    num_gpus: int = None,
    model=None,
    tokenizer=None,
):
    """
    Test if model actually learned the knowledge using batch inference.

    Args:
        model_path: Path to model (used if model/tokenizer not provided)
        samples: List of samples to test
        num_trials: Number of generation trials per sample
        inference_batch_size: Batch size for inference
        num_gpus: Number of GPUs (for multi-GPU inference when loading from path)
        model: Pre-loaded model (optional, avoids reloading)
        tokenizer: Pre-loaded tokenizer (optional, avoids reloading)

    Returns accuracy on the QA samples.
    """
    print(f"\nTesting knowledge acquisition on {len(samples)} samples...")

    # Filter samples with valid answers
    valid_samples = []
    for sample in samples:
        gold_answers = sample.get("normalized_answers", sample.get("answers", []))
        if gold_answers:
            valid_samples.append(sample)

    if not valid_samples:
        return {"accuracy": 0, "correct": 0, "total": 0}

    print(f"Building {len(valid_samples)} × {num_trials} = {len(valid_samples) * num_trials} prompts...")

    # Build all prompts at once: samples × num_trials
    all_prompts = []
    for sample in valid_samples:
        prompt = f"Question: {sample['question']}\nAnswer:"
        all_prompts.extend([prompt] * num_trials)

    # Use provided model or create inference instance
    if model is not None and tokenizer is not None:
        # Use provided model directly (single GPU, already loaded)
        print("Using pre-loaded model for inference...")
        model.eval()
        all_responses = []

        # Process in batches
        for i in range(0, len(all_prompts), inference_batch_size):
            batch_prompts = all_prompts[i:i + inference_batch_size]

            # Tokenize batch with LEFT padding for generation
            tokenizer.padding_side = "left"
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
            for j, output in enumerate(outputs):
                response = tokenizer.decode(
                    output[inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                all_responses.append(response)

            if (i // inference_batch_size + 1) % 10 == 0:
                print(f"  Processed {min(i + inference_batch_size, len(all_prompts))}/{len(all_prompts)} prompts")

        # No cleanup needed - caller manages the model
        should_cleanup = False
    else:
        # Create inference instance (auto multi-GPU)
        inference = create_inference(
            model_name=model_path,
            inference_batch_size=inference_batch_size,
            temperature=1.0,
            num_gpus=num_gpus,
        )

        # Batch generate all responses at once
        print("Running batch inference...")
        all_responses = inference.generate_batch(all_prompts)
        should_cleanup = True

        # Clean up inference - ensure complete release before any subsequent operations
        if hasattr(inference, 'shutdown'):
            inference.shutdown()
        del inference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(2.0)  # Wait for GPU memory to be fully released

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
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Knowledge Learning")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")

    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--input", type=str,
                        default=str(project_root / "data/step1_responses.jsonl"),
                        help="Input file with questions and answers (from Phase 1)")
    parser.add_argument("--output_dir", type=str,
                        default=str(project_root / "outputs/phase2_knowledge"))
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of epochs (default: 15 for adaptive)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="1e-4 for LoRA, 1e-5 for full fine-tuning")
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--no_lora", action="store_true",
                        help="Disable LoRA for full fine-tuning")
    parser.add_argument("--no_merge", action="store_true",
                        help="Don't merge adapter into base model")
    parser.add_argument("--test_samples", type=int, default=100,
                        help="Number of samples to test knowledge acquisition")
    parser.add_argument("--adaptive", action="store_true", default=True,
                        help="Use adaptive training (train each sample until learned)")
    parser.add_argument("--max_steps_per_sample", type=int, default=10,
                        help="Max training steps per sample in adaptive mode")
    parser.add_argument("--filter_ability", type=str, nargs="+", default=None,
                        help="Only train samples with these abilities (e.g., --filter_ability cannot uncertain)")
    parser.add_argument("--skip_correct", action="store_true", default=True,
                        help="Skip samples model already answers correctly (test before each epoch)")

    # Pipeline integration
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name for pipeline integration")

    # GPU params
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs to use for inference (default: all available)")

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

    # Initialize pipeline if experiment name provided
    pipeline = None
    if args.experiment:
        pipeline = MultiPhasePipeline(
            experiment_name=args.experiment,
            base_model=args.model,
            output_dir=str(project_root / "experiments")
        )
        output_dir = pipeline.get_phase_output_dir("phase2_knowledge")

    # Step 2.1: Load data and build QA dataset
    print(f"Loading data from {args.input}")
    samples = load_from_jsonl(args.input)
    print(f"Loaded {len(samples)} samples")

    print("Building QA training dataset...")
    qa_data = build_qa_dataset(samples)
    print(f"Created {len(qa_data)} QA training samples")

    # Save QA data
    qa_data_path = output_dir / "qa_training_data.jsonl"
    save_to_jsonl(qa_data, str(qa_data_path))
    print(f"Saved QA data to {qa_data_path}")

    # Show examples
    print("\n--- Example QA samples ---")
    for i, sample in enumerate(qa_data[:3]):
        user_msg = sample["messages"][1]["content"]
        assistant_msg = sample["messages"][2]["content"]
        print(f"\n[{i+1}] Question: {user_msg[:60]}...")
        print(f"    Answer: {assistant_msg}")

    # Step 2.2: Baseline evaluation BEFORE training (GPU is free now)
    train_test_samples = samples[:args.test_samples]
    val_test_samples = load_triviaqa(split="validation", num_samples=args.test_samples)

    if is_main_process():
        print(f"\n--- Baseline evaluation on TRAIN split ---")
        print(f"Testing on {len(train_test_samples)} samples (train split)")
        print("\nBefore training (base model) on TRAIN:")
    before_train = test_knowledge_acquisition(
        args.model, train_test_samples,
        inference_batch_size=args.inference_batch_size,
        num_gpus=args.num_gpus
    )
    if is_main_process():
        print(f"Accuracy: {before_train['accuracy']*100:.1f}% ({before_train['correct']}/{before_train['total']})")

    if is_main_process():
        print(f"\n--- Baseline evaluation on VALIDATION split ---")
        print(f"Testing on {len(val_test_samples)} samples (validation split)")
        print("\nBefore training (base model) on VALIDATION:")
    before_val = test_knowledge_acquisition(
        args.model, val_test_samples,
        inference_batch_size=args.inference_batch_size,
        num_gpus=args.num_gpus
    )
    if is_main_process():
        print(f"Accuracy: {before_val['accuracy']*100:.1f}% ({before_val['correct']}/{before_val['total']})")

    # Step 2.3: Setup model and train
    if is_main_process():
        print(f"\nSetting up model: {args.model}")
        print(f"Training mode: {'Full fine-tuning' if args.no_lora else 'LoRA'}")
        print(f"Adaptive training: {args.adaptive}")
        print(f"DDP: {args.ddp}")
    model, tokenizer = setup_model_for_training(
        args.model,
        use_lora=not args.no_lora,
        ddp=args.ddp,
        local_rank=local_rank,
    )

    adapter_path = output_dir / "knowledge"

    if args.adaptive:
        # Prepare samples with question and answers for adaptive training
        adaptive_samples = []
        for sample in qa_data:
            # Extract question from messages
            question = sample["messages"][1]["content"]  # User message is the question
            answer = sample["messages"][2]["content"]  # Assistant message is the answer
            adaptive_samples.append({
                "messages": sample["messages"],
                "question": question,
                "answers": [answer],
                "normalized_answers": [answer],
                "original_ability": sample.get("original_ability", ""),  # Preserve ability for filtering
            })

        if args.ddp:
            # DDP adaptive training
            from src.ddp_adaptive_trainer import DDPAdaptiveKnowledgeTrainer, is_main_process, cleanup_ddp

            if is_main_process():
                print(f"\nUsing DDP adaptive training")
                print(f"Epochs: {args.epochs}")
                print(f"Skip already correct: {args.skip_correct}")
                if args.filter_ability:
                    print(f"Filtering to abilities: {args.filter_ability}")

            trainer = DDPAdaptiveKnowledgeTrainer(
                model=model,
                tokenizer=tokenizer,
                learning_rate=args.lr,
                local_rank=local_rank,
            )

            stats = trainer.train_dataset(
                adaptive_samples,
                num_epochs=args.epochs,
                filter_by_ability=args.filter_ability,
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
            from src.adaptive_trainer import AdaptiveKnowledgeTrainer

            print(f"\nUsing adaptive training (max {args.max_steps_per_sample} steps per sample)")
            print(f"Epochs: {args.epochs}")

            trainer = AdaptiveKnowledgeTrainer(
                model=model,
                tokenizer=tokenizer,
                learning_rate=args.lr,
                max_steps_per_sample=args.max_steps_per_sample,
            )

            print(f"\nTraining on {len(adaptive_samples)} samples...")
            if args.filter_ability:
                print(f"Filtering to abilities: {args.filter_ability}")
            print(f"Skip already correct: {args.skip_correct}")

            stats = trainer.train_dataset(
                adaptive_samples,
                num_epochs=args.epochs,
                filter_by_ability=args.filter_ability,
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
        datasets = prepare_dataset_for_training(qa_data, tokenizer)
        print(f"Train: {len(datasets['train'])}, Val: {len(datasets['validation'])}")

        print(f"\nTraining knowledge adapter...")
        print(f"Output: {adapter_path}")

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
        # Only main process continues with evaluation and merging
        if not is_main_process():
            cleanup_ddp()
            return

    if is_main_process():
        print(f"\nKnowledge model saved to {adapter_path}")

    # Step 2.4: Merge adapter into base model (or use full fine-tuned model directly)
    merged_path = None
    eval_model = None
    eval_tokenizer = None

    if args.no_lora:
        # Full fine-tuning: the trained model is already complete, use it directly
        merged_path = adapter_path  # adapter_path contains the full model
        eval_model = model  # Reuse training model
        eval_tokenizer = tokenizer
        if is_main_process():
            print(f"\nFull fine-tuning: Model saved directly to {merged_path}")
    elif not args.no_merge:
        merged_path = output_dir / "base_with_knowledge"
        if is_main_process():
            # Need to clean up training model before merging to free GPU memory
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            time.sleep(1.0)
            print("Cleaned up training model from GPU memory")

            print(f"\nMerging adapter into base model...")
            merge_adapter_into_base(
                args.model,
                str(adapter_path),
                str(merged_path)
            )
            print(f"Merged model saved to {merged_path}")

            # Load merged model for evaluation
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("Loading merged model for evaluation...")
            eval_model = AutoModelForCausalLM.from_pretrained(
                str(merged_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            eval_tokenizer = AutoTokenizer.from_pretrained(str(merged_path), trust_remote_code=True)
            if eval_tokenizer.pad_token is None:
                eval_tokenizer.pad_token = eval_tokenizer.eos_token

    # Step 2.5: Test knowledge acquisition using pre-loaded model
    test_model_path = str(merged_path) if merged_path else args.model

    if is_main_process() and eval_model is not None:
        print(f"\n--- After-training evaluation on TRAIN split ---")
        print(f"Testing on {len(train_test_samples)} samples (train split)")
        print("\nAfter training (with knowledge) on TRAIN:")
        after_train = test_knowledge_acquisition(
            test_model_path, train_test_samples,
            inference_batch_size=args.inference_batch_size,
            num_gpus=args.num_gpus,
            model=eval_model, tokenizer=eval_tokenizer
        )
        train_improvement = after_train['accuracy'] - before_train['accuracy']
        print(f"Accuracy: {after_train['accuracy']*100:.1f}% ({after_train['correct']}/{after_train['total']})")
        print(f"Train Improvement: {train_improvement*100:+.1f}%")

        print(f"\n--- After-training evaluation on VALIDATION split ---")
        print(f"Testing on {len(val_test_samples)} samples (validation split - held-out)")
        print("\nAfter training (with knowledge) on VALIDATION:")
        after_val = test_knowledge_acquisition(
            test_model_path, val_test_samples,
            inference_batch_size=args.inference_batch_size,
            num_gpus=args.num_gpus,
            model=eval_model, tokenizer=eval_tokenizer
        )
        val_improvement = after_val['accuracy'] - before_val['accuracy']
        print(f"Accuracy: {after_val['accuracy']*100:.1f}% ({after_val['correct']}/{after_val['total']})")
        print(f"Validation Improvement: {val_improvement*100:+.1f}%")

        # Clean up evaluation model
        del eval_model
        del eval_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Cleaned up evaluation model from GPU memory")
    else:
        # Fallback: use regular inference (should not happen normally)
        after_train = {"accuracy": 0, "correct": 0, "total": 0}
        after_val = {"accuracy": 0, "correct": 0, "total": 0}
        train_improvement = 0
        val_improvement = 0

    # Record to pipeline if available
    if pipeline and is_main_process():
        pipeline.record_phase_result(
            phase_name="phase2_knowledge",
            status="completed",
            metrics={
                "qa_samples": len(qa_data),
                "train_before_accuracy": before_train['accuracy'],
                "train_after_accuracy": after_train['accuracy'],
                "train_improvement": train_improvement,
                "val_before_accuracy": before_val['accuracy'],
                "val_after_accuracy": after_val['accuracy'],
                "val_improvement": val_improvement,
            },
            output_paths={
                "qa_data": str(qa_data_path),
                "knowledge": str(adapter_path),
                "base_with_knowledge": str(merged_path) if merged_path else "",
            }
        )
        pipeline.print_summary()

    if is_main_process():
        print("\n" + "=" * 60)
        print("Phase 2 (Knowledge Learning) completed!")
        print("=" * 60)
        if merged_path:
            print(f"Next step: Use '{merged_path}' as base model for Phase 3")

    # Final DDP cleanup
    if args.ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
