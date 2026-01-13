"""
Step 3: Train model for metacognition.
"""

import argparse
import sys
sys.path.append("..")

from src.dataset_builder import load_from_jsonl, prepare_dataset_for_training
from src.trainer import setup_model_for_training, train_metacognition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--input", type=str, default="../data/step2_training_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="../outputs/metacog")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA")
    args = parser.parse_args()

    print(f"Loading training data from {args.input}")
    training_data = load_from_jsonl(args.input)
    print(f"Loaded {len(training_data)} samples")

    print(f"Setting up model: {args.model}")
    model, tokenizer = setup_model_for_training(
        args.model,
        use_lora=not args.no_lora,
    )

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
    )

    print(f"Model saved to {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
