"""
Data-parallel inference for collecting responses and testing judgments.

Each GPU loads a full model copy and processes a shard of data.

Usage:
    # Collect responses (determine ability labels)
    torchrun --nproc_per_node=8 scripts/inference_ddp.py \
        --mode collect \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output_dir experiments/judgment \
        --num_samples 5000

    # Test judgment accuracy
    torchrun --nproc_per_node=8 scripts/inference_ddp.py \
        --mode test \
        --model experiments/judgment/epoch_1 \
        --input experiments/judgment/responses.jsonl \
        --output_dir experiments/judgment
"""

import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.data_loader import load_triviaqa, format_question_prompt
from src.dataset_builder import load_from_jsonl, save_to_jsonl
from src.evaluator import is_correct, classify_ability, classify_ability_binary
from src.label_generator import get_system_prompt


def collect_responses_binary(samples, model, tokenizer, batch_size=8, max_new_tokens=64,
                             local_rank=0, world_size=1, show_progress=True):
    """Collect responses with temperature=0, single trial."""
    # Shard data across ranks
    shard_size = (len(samples) + world_size - 1) // world_size
    start_idx = local_rank * shard_size
    end_idx = min(start_idx + shard_size, len(samples))
    local_samples = samples[start_idx:end_idx]

    results = []
    model.eval()

    iterator = range(0, len(local_samples), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Rank {local_rank} collecting responses")

    for i in iterator:
        batch = local_samples[i:i + batch_size]
        prompts = [format_question_prompt(s["question"]) for s in batch]

        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted)

        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, (sample, output) in enumerate(zip(batch, outputs)):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()

            gold_answers = sample.get("normalized_answers", sample.get("answers", []))
            correct = is_correct(response, gold_answers)
            ability = classify_ability_binary(correct)

            result = sample.copy()
            result["response"] = response
            result["correct"] = correct
            result["ability"] = ability
            result["_original_idx"] = start_idx + i + j
            results.append(result)

    return results


def collect_responses_uncertainty(samples, model, tokenizer, num_trials=10, batch_size=8,
                                  max_new_tokens=64, temperature=0.7,
                                  local_rank=0, world_size=1, show_progress=True):
    """Collect responses with temperature>0, multiple trials."""
    # Shard data across ranks
    shard_size = (len(samples) + world_size - 1) // world_size
    start_idx = local_rank * shard_size
    end_idx = min(start_idx + shard_size, len(samples))
    local_samples = samples[start_idx:end_idx]

    results = []
    model.eval()

    iterator = local_samples
    if show_progress:
        iterator = tqdm(iterator, desc=f"Rank {local_rank} collecting responses")

    for idx, sample in enumerate(iterator):
        prompt = format_question_prompt(sample["question"])
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(
            [formatted] * num_trials,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )

        responses = []
        correct_count = 0
        gold_answers = sample.get("normalized_answers", sample.get("answers", []))

        for k, output in enumerate(outputs):
            input_len = inputs["input_ids"][k].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            responses.append(response)
            if is_correct(response, gold_answers):
                correct_count += 1

        ability = classify_ability(correct_count, num_trials)

        result = sample.copy()
        result["responses"] = responses
        result["correct_count"] = correct_count
        result["ability"] = ability
        result["_original_idx"] = start_idx + idx
        results.append(result)

    return results


def eval_qa_accuracy(samples, model, tokenizer, batch_size=8, max_new_tokens=64,
                     local_rank=0, world_size=1, show_progress=True):
    """Evaluate model's QA accuracy on given samples."""
    # Shard data across ranks
    shard_size = (len(samples) + world_size - 1) // world_size
    start_idx = local_rank * shard_size
    end_idx = min(start_idx + shard_size, len(samples))
    local_samples = samples[start_idx:end_idx]

    results = []
    model.eval()

    iterator = range(0, len(local_samples), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Rank {local_rank} evaluating QA")

    for i in iterator:
        batch = local_samples[i:i + batch_size]
        prompts = [format_question_prompt(s["question"]) for s in batch]

        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted)

        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, (sample, output) in enumerate(zip(batch, outputs)):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()

            gold_answers = sample.get("normalized_answers", sample.get("answers", []))
            correct = is_correct(response, gold_answers)

            result = sample.copy()
            result["eval_response"] = response
            result["eval_correct"] = correct
            result["_original_idx"] = start_idx + i + j
            results.append(result)

    return results


def parse_judgment_binary(response: str) -> str:
    """Parse judgment from response (binary mode: yes/no)."""
    response = response.lower()
    if "\\boxed{yes}" in response or "boxed{yes}" in response:
        return "can"
    elif "\\boxed{no}" in response or "boxed{no}" in response:
        return "cannot"
    # Fallback: check for plain yes/no
    elif response.strip() == "yes":
        return "can"
    elif response.strip() == "no":
        return "cannot"
    # Default to cannot if unclear
    return "cannot"


def parse_judgment_uncertainty(response: str) -> str:
    """Parse judgment from response (uncertainty mode: yes/uncertain/no)."""
    response = response.lower()
    if "\\boxed{yes}" in response or "boxed{yes}" in response:
        return "can"
    elif "\\boxed{uncertain}" in response or "boxed{uncertain}" in response:
        return "uncertain"
    elif "\\boxed{no}" in response or "boxed{no}" in response:
        return "cannot"
    # Fallback
    elif response.strip() == "yes":
        return "can"
    elif response.strip() == "uncertain":
        return "uncertain"
    elif response.strip() == "no":
        return "cannot"
    return "cannot"


def test_judgment_accuracy(samples, model, tokenizer, batch_size=8,
                           label_mode="binary", local_rank=0, world_size=1, show_progress=True):
    """Test model's judgment accuracy with confidence scores for AUROC."""
    # Shard data across ranks
    shard_size = (len(samples) + world_size - 1) // world_size
    start_idx = local_rank * shard_size
    end_idx = min(start_idx + shard_size, len(samples))
    local_samples = samples[start_idx:end_idx]

    results = []
    model.eval()

    # Get system prompt based on mode
    system_prompt = get_system_prompt(label_mode)
    parse_fn = parse_judgment_binary if label_mode == "binary" else parse_judgment_uncertainty

    # Get token IDs for yes/no (for probability computation)
    yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[-1]
    no_token_id = tokenizer.encode("no", add_special_tokens=False)[-1]

    iterator = range(0, len(local_samples), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Rank {local_rank} testing judgments")

    for i in iterator:
        batch = local_samples[i:i + batch_size]

        prompts = []
        for s in batch:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Can you answer this question correctly?\n\nQuestion: {s['question']}"},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            # First, get logits for the first generated token to compute confidence
            model_outputs = model(**inputs)
            logits = model_outputs.logits  # [batch_size, seq_len, vocab_size]

            # Compute yes probability for each sample
            batch_probs = []
            for j in range(len(batch)):
                # Find the last non-padding position
                attention_mask = inputs["attention_mask"][j]
                last_pos = attention_mask.sum() - 1

                # Get logits at last position (predicts first output token)
                next_token_logits = logits[j, last_pos, :]

                # Compute softmax probability for yes vs no
                yes_no_logits = torch.tensor([
                    next_token_logits[yes_token_id].item(),
                    next_token_logits[no_token_id].item()
                ])
                yes_no_probs = torch.softmax(yes_no_logits, dim=0)
                yes_prob = yes_no_probs[0].item()
                batch_probs.append(yes_prob)

            # Generate full response for parsing
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, (sample, output) in enumerate(zip(batch, outputs)):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()

            predicted = parse_fn(response)
            yes_prob = batch_probs[j]

            # Debug: print first few responses
            if local_rank == 0 and i == 0 and j < 3:
                print(f"[DEBUG] Response: '{response}' -> {predicted}, yes_prob: {yes_prob:.4f}")

            ground_truth = sample["ability"]
            judgment_correct = (predicted == ground_truth)

            result = sample.copy()
            result["predicted_judgment"] = predicted
            result["judgment_correct"] = judgment_correct
            result["yes_prob"] = yes_prob  # Confidence score for AUROC
            result["_original_idx"] = start_idx + i + j
            results.append(result)

    return results


def merge_results(output_dir, world_size, prefix):
    """Merge results from all ranks."""
    all_results = []
    for rank in range(world_size):
        rank_file = output_dir / f"{prefix}_rank{rank}.jsonl"
        if rank_file.exists():
            rank_results = load_from_jsonl(str(rank_file))
            all_results.extend(rank_results)
            rank_file.unlink()  # Cleanup

    # Sort by original index
    all_results.sort(key=lambda x: x.get("_original_idx", 0))
    for r in all_results:
        r.pop("_original_idx", None)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["collect", "test", "eval_qa"],
                        help="Mode: collect (responses), test (judgments), or eval_qa (QA accuracy)")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    # Data options
    parser.add_argument("--input", type=str, default=None, help="Input JSONL file")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")

    # Collect mode options
    parser.add_argument("--label_mode", type=str, default="binary", choices=["binary", "uncertainty"])
    parser.add_argument("--num_trials", type=int, default=10, help="Trials for uncertainty mode")
    parser.add_argument("--temperature", type=float, default=0.7)

    # Common options
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_file", type=str, default=None, help="Output filename (default: auto)")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline file for comparison (epoch 0)")

    args = parser.parse_args()

    # Get distributed info
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    # Initialize distributed
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model on this GPU
    device = f"cuda:{local_rank}"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # Load data
    if args.input:
        samples = load_from_jsonl(args.input)
    else:
        samples = load_triviaqa(split=args.split, num_samples=args.num_samples)

    # Run inference
    if args.mode == "collect":
        if args.label_mode == "binary":
            local_results = collect_responses_binary(
                samples, model, tokenizer,
                batch_size=args.batch_size,
                local_rank=local_rank,
                world_size=world_size,
                show_progress=is_main
            )
        else:
            local_results = collect_responses_uncertainty(
                samples, model, tokenizer,
                num_trials=args.num_trials,
                batch_size=args.batch_size,
                temperature=args.temperature,
                local_rank=local_rank,
                world_size=world_size,
                show_progress=is_main
            )
        prefix = "responses"
        default_output = "responses.jsonl"

    elif args.mode == "test":
        local_results = test_judgment_accuracy(
            samples, model, tokenizer,
            batch_size=args.batch_size,
            label_mode=args.label_mode,
            local_rank=local_rank,
            world_size=world_size,
            show_progress=is_main
        )
        prefix = "tested"
        default_output = "tested.jsonl"

    else:  # eval_qa
        local_results = eval_qa_accuracy(
            samples, model, tokenizer,
            batch_size=args.batch_size,
            local_rank=local_rank,
            world_size=world_size,
            show_progress=is_main
        )
        prefix = "eval_qa"
        default_output = "eval_qa.jsonl"

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Save local results
    local_file = output_dir / f"{prefix}_rank{local_rank}.jsonl"
    save_to_jsonl(local_results, str(local_file))

    # Sync all ranks
    if world_size > 1:
        dist.barrier()

    # Merge on main process
    if is_main:
        all_results = merge_results(output_dir, world_size, prefix)

        output_file = args.output_file or default_output
        output_path = output_dir / output_file
        save_to_jsonl(all_results, str(output_path))

        # Print stats
        if args.mode == "collect":
            # Print QA accuracy (can = correct, cannot = incorrect)
            correct = sum(1 for r in all_results if r["ability"] == "can")
            current_acc = correct / len(all_results) * 100

            # Load baseline for comparison
            baseline_acc = None
            if args.baseline and Path(args.baseline).exists():
                baseline_results = load_from_jsonl(args.baseline)
                baseline_correct = sum(1 for r in baseline_results if r.get("ability") == "can")
                baseline_acc = baseline_correct / len(baseline_results) * 100

            if baseline_acc is not None:
                diff = current_acc - baseline_acc
                diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
                print(f"\nQA accuracy: {correct}/{len(all_results)} ({current_acc:.1f}%)  [baseline: {baseline_acc:.1f}%, diff: {diff_str}]")
            else:
                print(f"\nQA accuracy: {correct}/{len(all_results)} ({current_acc:.1f}%)")

        elif args.mode == "test":
            correct = sum(1 for r in all_results if r["judgment_correct"])
            current_acc = correct / len(all_results) * 100

            # Load baseline for comparison
            baseline_acc = None
            if args.baseline and Path(args.baseline).exists():
                baseline_results = load_from_jsonl(args.baseline)
                baseline_correct = sum(1 for r in baseline_results if r["judgment_correct"])
                baseline_acc = baseline_correct / len(baseline_results) * 100

            if baseline_acc is not None:
                diff = current_acc - baseline_acc
                diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
                print(f"\nJudgment accuracy: {correct}/{len(all_results)} ({current_acc:.1f}%)  [baseline: {baseline_acc:.1f}%, diff: {diff_str}]")
            else:
                print(f"\nJudgment accuracy: {correct}/{len(all_results)} ({current_acc:.1f}%)")

            # Determine abilities based on actual data
            actual_abilities = set()
            for r in all_results:
                actual_abilities.add(r["ability"])
                actual_abilities.add(r["predicted_judgment"])

            # Use appropriate ability set
            if "uncertain" in actual_abilities:
                abilities = ["can", "uncertain", "cannot"]
            else:
                abilities = ["can", "cannot"]

            # Print confusion matrix
            print(f"\nConfusion Matrix:")
            col_width = 10
            header = "".join(f"{a:>{col_width}}" for a in abilities)
            print(f"{'':15} {'Predicted':^{len(header)}}")
            print(f"{'Ground Truth':15} {header}")
            print("-" * (15 + len(header) + 10))

            matrix = {gt: {pred: 0 for pred in abilities} for gt in abilities}

            for r in all_results:
                gt = r["ability"]
                pred = r["predicted_judgment"]
                if gt in matrix and pred in abilities:
                    matrix[gt][pred] += 1

            for gt in abilities:
                row = matrix[gt]
                total = sum(row.values())
                if total > 0:
                    row_str = "".join(f"{row[pred]:>{col_width}}" for pred in abilities)
                    print(f"{gt:15} {row_str}  (n={total})")

            print("-" * (15 + len(header) + 10))

            # Compute AUROC using yes_prob as confidence score
            # Ground truth: can=1 (positive), cannot=0 (negative)
            y_true = [1 if r["ability"] == "can" else 0 for r in all_results]
            y_scores = [r.get("yes_prob", 0.5) for r in all_results]

            # Only compute AUROC if we have both classes
            if len(set(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_scores)

                # Load baseline AUROC for comparison
                baseline_auroc = None
                if args.baseline and Path(args.baseline).exists():
                    baseline_results = load_from_jsonl(args.baseline)
                    if baseline_results and "yes_prob" in baseline_results[0]:
                        b_y_true = [1 if r["ability"] == "can" else 0 for r in baseline_results]
                        b_y_scores = [r.get("yes_prob", 0.5) for r in baseline_results]
                        if len(set(b_y_true)) > 1:
                            baseline_auroc = roc_auc_score(b_y_true, b_y_scores)

                if baseline_auroc is not None:
                    diff = auroc - baseline_auroc
                    diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
                    print(f"\nAUROC: {auroc:.4f}  [baseline: {baseline_auroc:.4f}, diff: {diff_str}]")
                else:
                    print(f"\nAUROC: {auroc:.4f}")
            else:
                print(f"\nAUROC: N/A (only one class present)")

        else:  # eval_qa
            correct = sum(1 for r in all_results if r["eval_correct"])
            current_acc = correct / len(all_results) * 100

            # Load baseline for comparison (supports both responses.jsonl and eval_qa.jsonl formats)
            baseline_acc = None
            if args.baseline and Path(args.baseline).exists():
                baseline_results = load_from_jsonl(args.baseline)
                # Check which field exists: "eval_correct" (eval_qa) or "ability" (collect)
                if baseline_results and "eval_correct" in baseline_results[0]:
                    baseline_correct = sum(1 for r in baseline_results if r["eval_correct"])
                else:
                    # responses.jsonl uses "ability" field (can = correct)
                    baseline_correct = sum(1 for r in baseline_results if r.get("ability") == "can")
                baseline_acc = baseline_correct / len(baseline_results) * 100

            if baseline_acc is not None:
                diff = current_acc - baseline_acc
                diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
                print(f"\nQA accuracy: {correct}/{len(all_results)} ({current_acc:.1f}%)  [baseline: {baseline_acc:.1f}%, diff: {diff_str}]")
            else:
                print(f"\nQA accuracy: {correct}/{len(all_results)} ({current_acc:.1f}%)")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
