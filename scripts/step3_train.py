"""
Step 3: Train model for metacognition (judgment training).

Supports two training modes:
- Standard: Fixed epochs with batch training
- Adaptive: Train each sample until model predicts correct ability
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_builder import load_from_jsonl, prepare_dataset_for_training
from src.trainer import setup_model_for_training, train_metacognition
from src.label_generator import SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--input", type=str, default=str(project_root / "data/step2_training_data.jsonl"))
    parser.add_argument("--output_dir", type=str, default=str(project_root / "outputs/metacog"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="1e-4 for LoRA, 1e-5 for full fine-tuning")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--adaptive", action="store_true", default=True,
                        help="Use adaptive training (train each sample until correct)")
    parser.add_argument("--max_steps_per_sample", type=int, default=10,
                        help="Max training steps per sample in adaptive mode")
    parser.add_argument("--skip_correct", action="store_true", default=True,
                        help="Skip samples model already judges correctly (test before each epoch)")
    args = parser.parse_args()

    print(f"Loading training data from {args.input}")
    training_data = load_from_jsonl(args.input)
    print(f"Loaded {len(training_data)} samples")

    print(f"Setting up model: {args.model}")
    print(f"Training mode: {'Full fine-tuning' if args.no_lora else 'LoRA'}")
    print(f"Adaptive training: {args.adaptive}")
    model, tokenizer = setup_model_for_training(
        args.model,
        use_lora=not args.no_lora,
    )

    if args.adaptive:
        # Adaptive training for judgment
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
        print(f"\nSaving model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        print(f"\nAdaptive training complete!")
        print(f"Final stats: {stats['per_epoch'][-1]}")

    else:
        # Standard training
        print("Preparing dataset...")
        datasets = prepare_dataset_for_training(training_data, tokenizer)
        print(f"Train: {len(datasets['train'])}, Val: {len(datasets['validation'])}")

        print("Starting training...")
        train_metacognition(
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            val_dataset=datasets["validation"],
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_lora=not args.no_lora,
        )

    print(f"Model saved to {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
