"""
Phase 2: Knowledge Learning

Train the model to learn factual knowledge by teaching it to answer questions correctly.
This is different from judgment training - here we teach knowledge, not metacognition.

Supports two training modes:
- Standard: Fixed epochs with batch training
- Adaptive: Train each sample until learned (tested after each step)

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


def test_knowledge_acquisition(
    model_path: str,
    samples: list,
    num_trials: int = 5,
    inference_batch_size: int = 16,
    multi_gpu: bool = False,
    num_gpus: int = None
):
    """
    Test if model actually learned the knowledge.

    Returns accuracy on the QA samples.
    """
    print(f"\nTesting knowledge acquisition on {len(samples)} samples...")
    print(f"Multi-GPU: {multi_gpu}")

    inference = create_inference(
        model_name=model_path,
        inference_batch_size=inference_batch_size,
        temperature=1.0,
        multi_gpu=multi_gpu,
        num_gpus=num_gpus,
    )

    correct_count = 0
    total = 0

    for sample in tqdm(samples, desc="Testing knowledge"):
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
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs (default: 2 for adaptive)")
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

    # Multi-GPU inference
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use multiple GPUs for inference (data parallelism)")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs to use (default: all available)")

    args = parser.parse_args()

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

    # Step 2.2: Setup model and train
    print(f"\nSetting up model: {args.model}")
    print(f"Training mode: {'Full fine-tuning' if args.no_lora else 'LoRA'}")
    print(f"Adaptive training: {args.adaptive}")
    model, tokenizer = setup_model_for_training(args.model, use_lora=not args.no_lora)

    adapter_path = output_dir / "knowledge"

    if args.adaptive:
        # Adaptive training: train each sample until learned
        from src.adaptive_trainer import AdaptiveKnowledgeTrainer

        print(f"\nUsing adaptive training (max {args.max_steps_per_sample} steps per sample)")
        print(f"Epochs: {args.epochs}")

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

    print(f"\nKnowledge model saved to {adapter_path}")

    # Step 2.3: Merge adapter into base model (or use full fine-tuned model directly)
    merged_path = None
    if args.no_lora:
        # Full fine-tuning: the trained model is already complete, use it directly
        merged_path = adapter_path  # adapter_path contains the full model
        print(f"\nFull fine-tuning: Model saved directly to {merged_path}")
    elif not args.no_merge:
        merged_path = output_dir / "base_with_knowledge"
        print(f"\nMerging adapter into base model...")
        merge_adapter_into_base(
            args.model,
            str(adapter_path),
            str(merged_path)
        )
        print(f"Merged model saved to {merged_path}")

    # Step 2.4: Test knowledge acquisition on both train and validation splits
    test_model_path = str(merged_path) if merged_path else args.model

    # Test on TRAIN split (verify model learned training data)
    print(f"\n--- Testing knowledge acquisition on TRAIN split ---")
    train_test_samples = samples[:args.test_samples]  # Use subset of training data
    print(f"Testing on {len(train_test_samples)} samples (train split)")

    print("\nBefore training (base model) on TRAIN:")
    before_train = test_knowledge_acquisition(
        args.model, train_test_samples,
        inference_batch_size=args.inference_batch_size,
        multi_gpu=args.multi_gpu,
        num_gpus=args.num_gpus
    )
    print(f"Accuracy: {before_train['accuracy']*100:.1f}% ({before_train['correct']}/{before_train['total']})")

    print("\nAfter training (with knowledge) on TRAIN:")
    after_train = test_knowledge_acquisition(
        test_model_path, train_test_samples,
        inference_batch_size=args.inference_batch_size,
        multi_gpu=args.multi_gpu,
        num_gpus=args.num_gpus
    )
    print(f"Accuracy: {after_train['accuracy']*100:.1f}% ({after_train['correct']}/{after_train['total']})")

    train_improvement = after_train['accuracy'] - before_train['accuracy']
    print(f"Train Improvement: {train_improvement*100:+.1f}%")

    # Test on VALIDATION split (test generalization)
    print(f"\n--- Testing knowledge acquisition on VALIDATION split ---")
    val_test_samples = load_triviaqa(split="validation", num_samples=args.test_samples)
    print(f"Testing on {len(val_test_samples)} samples (validation split - held-out)")

    print("\nBefore training (base model) on VALIDATION:")
    before_val = test_knowledge_acquisition(
        args.model, val_test_samples,
        inference_batch_size=args.inference_batch_size,
        multi_gpu=args.multi_gpu,
        num_gpus=args.num_gpus
    )
    print(f"Accuracy: {before_val['accuracy']*100:.1f}% ({before_val['correct']}/{before_val['total']})")

    print("\nAfter training (with knowledge) on VALIDATION:")
    after_val = test_knowledge_acquisition(
        test_model_path, val_test_samples,
        inference_batch_size=args.inference_batch_size,
        multi_gpu=args.multi_gpu,
        num_gpus=args.num_gpus
    )
    print(f"Accuracy: {after_val['accuracy']*100:.1f}% ({after_val['correct']}/{after_val['total']})")

    val_improvement = after_val['accuracy'] - before_val['accuracy']
    print(f"Validation Improvement: {val_improvement*100:+.1f}%")

    # Record to pipeline if available
    if pipeline:
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

    print("\n" + "=" * 60)
    print("Phase 2 (Knowledge Learning) completed!")
    print("=" * 60)
    if merged_path:
        print(f"Next step: Use '{merged_path}' as base model for Phase 3")


if __name__ == "__main__":
    main()
