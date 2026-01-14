"""
Phase 2: Knowledge Learning

Train the model to learn factual knowledge by teaching it to answer questions correctly.
This is different from judgment training - here we teach knowledge, not metacognition.

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
from src.inference import ModelInference
from src.evaluator import is_correct
from src.pipeline import MultiPhasePipeline


def test_knowledge_acquisition(
    model_path: str,
    samples: list,
    num_trials: int = 5,
    inference_batch_size: int = 16
):
    """
    Test if model actually learned the knowledge.

    Returns accuracy on the QA samples.
    """
    print(f"\nTesting knowledge acquisition on {len(samples)} samples...")

    inference = ModelInference(
        model_name=model_path,
        inference_batch_size=inference_batch_size,
        temperature=1.0,
    )

    correct_count = 0
    total = 0

    for sample in samples:
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
    parser.add_argument("--epochs", type=int, default=5,
                        help="More epochs for knowledge learning")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--no_merge", action="store_true",
                        help="Don't merge adapter into base model")
    parser.add_argument("--test_samples", type=int, default=100,
                        help="Number of samples to test knowledge acquisition")

    # Pipeline integration
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name for pipeline integration")

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
    model, tokenizer = setup_model_for_training(args.model, use_lora=True)

    print("Preparing dataset...")
    datasets = prepare_dataset_for_training(qa_data, tokenizer)
    print(f"Train: {len(datasets['train'])}, Val: {len(datasets['validation'])}")

    adapter_path = output_dir / "lora_knowledge"
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
    )

    print(f"\nKnowledge adapter saved to {adapter_path}")

    # Step 2.3: Merge adapter into base model
    merged_path = None
    if not args.no_merge:
        merged_path = output_dir / "base_with_knowledge"
        print(f"\nMerging adapter into base model...")
        merge_adapter_into_base(
            args.model,
            str(adapter_path),
            str(merged_path)
        )
        print(f"Merged model saved to {merged_path}")

    # Step 2.4: Test knowledge acquisition
    test_model_path = str(merged_path) if merged_path else args.model
    test_samples = samples[:args.test_samples]

    print(f"\n--- Testing knowledge acquisition ---")
    print(f"Testing on {len(test_samples)} samples (original data)")

    # Test before (base model)
    print("\nBefore training (base model):")
    before_results = test_knowledge_acquisition(
        args.model, test_samples,
        inference_batch_size=args.inference_batch_size
    )
    print(f"Accuracy: {before_results['accuracy']*100:.1f}% ({before_results['correct']}/{before_results['total']})")

    # Test after (knowledge model)
    print("\nAfter training (with knowledge):")
    after_results = test_knowledge_acquisition(
        test_model_path, test_samples,
        inference_batch_size=args.inference_batch_size
    )
    print(f"Accuracy: {after_results['accuracy']*100:.1f}% ({after_results['correct']}/{after_results['total']})")

    improvement = after_results['accuracy'] - before_results['accuracy']
    print(f"\nImprovement: {improvement*100:+.1f}%")

    # Record to pipeline if available
    if pipeline:
        pipeline.record_phase_result(
            phase_name="phase2_knowledge",
            status="completed",
            metrics={
                "qa_samples": len(qa_data),
                "before_accuracy": before_results['accuracy'],
                "after_accuracy": after_results['accuracy'],
                "improvement": improvement,
            },
            output_paths={
                "qa_data": str(qa_data_path),
                "lora_knowledge": str(adapter_path),
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
