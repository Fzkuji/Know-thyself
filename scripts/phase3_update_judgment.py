"""
Phase 3: Update Judgment with Knowledge

After the model learns knowledge in Phase 2, re-train judgment ability.
The model should now be more confident since it actually knows the answers.

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
from src.inference import ModelInference
from src.evaluator import evaluate_responses, classify_ability
from src.dataset_builder import load_from_jsonl, save_to_jsonl, prepare_dataset_for_training
from src.label_generator import build_training_dataset, SYSTEM_PROMPT
from src.trainer import setup_model_for_training, train_metacognition
from src.pipeline import MultiPhasePipeline
from tqdm import tqdm
import re


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
    inference_batch_size: int = 16
):
    """
    Collect responses using the knowledge-augmented model.
    Returns samples with responses, evaluation, and ability.
    """
    print(f"\nCollecting responses using model: {model_path}")
    print(f"Samples: {len(samples)}, Trials per sample: {num_trials}")

    inference = ModelInference(
        model_name=model_path,
        inference_batch_size=inference_batch_size,
        temperature=1.0,
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

    return results


def evaluate_judgment(
    model_path: str,
    samples: list,
    adapter_path: str = None,
    inference_batch_size: int = 16
):
    """
    Evaluate judgment accuracy of the model.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"\nEvaluating judgment accuracy...")
    print(f"Model: {model_path}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    results = []
    abilities = ["can", "uncertain", "cannot"]

    for sample in tqdm(samples, desc="Evaluating judgment"):
        question = sample["question"]
        actual_ability = sample["ability"]

        # Build prompt using chat template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Can you answer this question correctly?\n\nQuestion: {question}"}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = parse_ability_prediction(response)

        results.append({
            "question": question,
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

    # Evaluation
    parser.add_argument("--test_samples", type=int, default=100)

    # Pipeline integration
    parser.add_argument("--experiment", type=str, default=None)

    args = parser.parse_args()

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
        inference_batch_size=args.inference_batch_size
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
    print(f"\nSetting up model for judgment training...")
    print(f"Base: {args.base_model}")
    print(f"Training mode: {'Full fine-tuning' if args.no_lora else 'LoRA'}")
    print(f"Adaptive training: {args.adaptive}")
    model, tokenizer = setup_model_for_training(args.base_model, use_lora=not args.no_lora)

    adapter_path = output_dir / "judgment_v2"

    if args.adaptive:
        # Adaptive training for judgment
        from src.adaptive_trainer import AdaptiveJudgmentTrainer

        print(f"\nUsing adaptive training (max {args.max_steps_per_sample} steps per sample)")
        print(f"Epochs: {args.epochs}")

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

    print(f"Judgment model saved to {adapter_path}")

    # Step 3.4: Final evaluation on both train and validation splits
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # ===== Evaluate on TRAIN split =====
    print("\n--- Evaluation on TRAIN split (verify learning) ---")
    train_test_samples = new_samples[:args.test_samples]
    train_dist = {}
    for s in train_test_samples:
        ability = s.get("ability", "unknown")
        train_dist[ability] = train_dist.get(ability, 0) + 1
    print(f"Train test ability distribution: {train_dist}")

    print("\nBefore judgment v2 training (knowledge model only) on TRAIN:")
    before_train_eval = evaluate_judgment(
        model_path=args.base_model,
        samples=train_test_samples,
        inference_batch_size=args.inference_batch_size
    )
    print(f"Exact Match: {before_train_eval['exact_match_rate']*100:.1f}%")

    print("\nAfter judgment v2 training on TRAIN:")
    if args.no_lora:
        # Full fine-tuning: model is saved directly in adapter_path
        after_train_eval = evaluate_judgment(
            model_path=str(adapter_path),
            samples=train_test_samples,
            adapter_path=None,
            inference_batch_size=args.inference_batch_size
        )
    else:
        # LoRA: use base model + adapter
        after_train_eval = evaluate_judgment(
            model_path=args.base_model,
            samples=train_test_samples,
            adapter_path=str(adapter_path),
            inference_batch_size=args.inference_batch_size
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
        inference_batch_size=args.inference_batch_size
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
        inference_batch_size=args.inference_batch_size
    )
    print(f"Exact Match: {before_val_eval['exact_match_rate']*100:.1f}%")

    print("\nAfter judgment v2 training on VALIDATION:")
    if args.no_lora:
        # Full fine-tuning: model is saved directly in adapter_path
        after_val_eval = evaluate_judgment(
            model_path=str(adapter_path),
            samples=val_test_samples,
            adapter_path=None,
            inference_batch_size=args.inference_batch_size
        )
    else:
        # LoRA: use base model + adapter
        after_val_eval = evaluate_judgment(
            model_path=args.base_model,
            samples=val_test_samples,
            adapter_path=str(adapter_path),
            inference_batch_size=args.inference_batch_size
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
            },
            output_paths={
                "responses": str(responses_path),
                "training_data": str(training_data_path),
                "judgment_v2": str(adapter_path),
            }
        )
        pipeline.print_summary()

    print("\n" + "=" * 60)
    print("Phase 3 (Update Judgment) completed!")
    print("=" * 60)
    print(f"\nFinal model: {args.base_model} + {adapter_path}")


if __name__ == "__main__":
    main()
