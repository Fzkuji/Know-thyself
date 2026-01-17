"""
Run Multi-phase Pipeline with DDP Support (Refactored)

DDP version of the three-phase training workflow. Uses a unified ModelManager
to handle model lifecycle across phases, avoiding repeated load/unload cycles.

Key design:
- ModelManager: Centralized model lifecycle management
- Phase functions: Only handle data prep, training logic, and evaluation
- Main loop: Coordinates model state and phase transitions

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
from datetime import datetime, timedelta
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
import gc

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


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    """Initialize distributed training with extended timeout for evaluation phases."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # Set very long timeout (24 hours) to allow single-rank evaluation without timeout
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
        return local_rank
    return 0


def cleanup_ddp():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============== Phase Results: Standardized Output ==============

def compute_confusion_matrix(pred_abilities: list, actual_abilities: list) -> dict:
    """
    Compute 3x3 confusion matrix for judgment evaluation.

    Returns:
        Dict with confusion matrix and per-class metrics
    """
    labels = ["can", "uncertain", "cannot"]
    matrix = {actual: {pred: 0 for pred in labels} for actual in labels}

    for pred, actual in zip(pred_abilities, actual_abilities):
        if actual in matrix and pred in labels:
            matrix[actual][pred] += 1

    # Compute per-class metrics
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class[label] = {
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100,
            "support": sum(matrix[label].values()),
        }

    return {
        "matrix": matrix,
        "per_class": per_class,
    }


def print_confusion_matrix(confusion: dict, indent: str = ""):
    """Print confusion matrix in a nice format."""
    matrix = confusion["matrix"]
    per_class = confusion["per_class"]
    labels = ["can", "uncertain", "cannot"]

    print(f"{indent}Confusion Matrix:")
    print(f"{indent}                    Predicted")
    print(f"{indent}                " + "".join(f"{l:>10}" for l in labels))
    print(f"{indent}           +" + "-" * 30)

    for actual in labels:
        row = f"{indent}Actual {actual:>8} |"
        for pred in labels:
            count = matrix[actual][pred]
            row += f"{count:>10}"
        print(row)

    print(f"\n{indent}Per-class Metrics:")
    print(f"{indent}{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"{indent}" + "-" * 52)
    for label in labels:
        m = per_class[label]
        print(f"{indent}{label:<12} {m['precision']:>9.1f}% {m['recall']:>9.1f}% {m['f1']:>9.1f}% {m['support']:>10}")


def print_stage_block(stage_name: str, train_data: dict, val_data: dict, show_qa: bool = True, show_judgment: bool = True):
    """
    Print a single stage block (Pre-trained Model or After Phase X Training).

    Args:
        stage_name: Name of the stage (e.g., "Pre-trained Model", "After Phase 1 Training")
        train_data: Dict with train split metrics
        val_data: Dict with validation split metrics
        show_qa: Whether to show QA ability section
        show_judgment: Whether to show Judgment ability section
    """
    print(f"\n  ┌─ {stage_name} " + "─" * (55 - len(stage_name)))
    print("  │")

    # QA Ability
    if show_qa and (train_data.get('qa_accuracy') is not None or val_data.get('qa_accuracy') is not None):
        print("  │  QA Ability:")
        if train_data.get('qa_accuracy') is not None:
            print(f"  │    Train:      {train_data['qa_accuracy']:5.1f}% ({train_data.get('qa_correct', '?')}/{train_data.get('qa_total', '?')})")
        if val_data.get('qa_accuracy') is not None:
            print(f"  │    Validation: {val_data['qa_accuracy']:5.1f}% ({val_data.get('qa_correct', '?')}/{val_data.get('qa_total', '?')})")
        print("  │")

    # Judgment Ability
    if show_judgment and (train_data.get('exact_match_rate') is not None or val_data.get('exact_match_rate') is not None):
        print("  │  Judgment Ability (Metacognition):")

        # Train split
        if train_data.get('exact_match_rate') is not None:
            print(f"  │    [Train] Exact Match: {train_data['exact_match_rate']:5.1f}%")
            if 'pred_can' in train_data:
                print(f"  │      Predicted: can={train_data['pred_can']}, uncertain={train_data['pred_uncertain']}, cannot={train_data['pred_cannot']}")
                print(f"  │      Actual:    can={train_data['actual_can']}, uncertain={train_data['actual_uncertain']}, cannot={train_data['actual_cannot']}")
            if 'confusion' in train_data:
                print("  │")
                print_confusion_matrix(train_data['confusion'], indent="  │      ")

        print("  │")

        # Validation split
        if val_data.get('exact_match_rate') is not None:
            print(f"  │    [Validation] Exact Match: {val_data['exact_match_rate']:5.1f}%")
            if 'pred_can' in val_data:
                print(f"  │      Predicted: can={val_data['pred_can']}, uncertain={val_data['pred_uncertain']}, cannot={val_data['pred_cannot']}")
                print(f"  │      Actual:    can={val_data['actual_can']}, uncertain={val_data['actual_uncertain']}, cannot={val_data['actual_cannot']}")
            if 'confusion' in val_data:
                print("  │")
                print_confusion_matrix(val_data['confusion'], indent="  │      ")

        print("  │")

    print("  └" + "─" * 60)


def print_phase_summary(phase: int, phase_name: str, results: dict, model_name: str = ""):
    """
    Print standardized summary for a phase.

    Format:
    - Phase 1: Pre-trained Model + After Training + Comparison
    - Phase 2/3: Only After Training (no pre-trained comparison)

    Args:
        phase: Phase number (1, 2, or 3)
        phase_name: Human-readable phase name
        results: Dict containing all evaluation results
        model_name: Model path/name for display
    """
    print("\n" + "=" * 70)
    print(f"  PHASE {phase} SUMMARY: {phase_name}")
    print("=" * 70)

    if model_name:
        print(f"  Model: {model_name}")

    # Extract before/after data
    before_train = results.get('before_train', {})
    before_val = results.get('before_val', {})
    after_train = results.get('after_train', {})
    after_val = results.get('after_val', {})

    # Determine what to show
    has_before = bool(before_train or before_val)
    has_after = bool(after_train or after_val)
    has_qa = any(d.get('qa_accuracy') is not None for d in [before_train, before_val, after_train, after_val])
    has_judgment = any(d.get('exact_match_rate') is not None for d in [before_train, before_val, after_train, after_val])

    # Phase 1: Show both Pre-trained Model and After Training
    # Phase 2/3: Only show After Training (no pre-trained comparison needed)
    show_before = (phase == 1) and has_before

    # Print Pre-trained Model block (only for Phase 1)
    if show_before:
        print_stage_block(
            "Pre-trained Model",
            before_train, before_val,
            show_qa=has_qa, show_judgment=has_judgment
        )

    # Print After Training block
    if has_after:
        print_stage_block(
            f"After Phase {phase} Training",
            after_train, after_val,
            show_qa=has_qa, show_judgment=has_judgment
        )

    # Print Comparison summary (only for Phase 1)
    if show_before:
        before_train_em = before_train.get('exact_match_rate')
        after_train_em = after_train.get('exact_match_rate')
        before_val_em = before_val.get('exact_match_rate')
        after_val_em = after_val.get('exact_match_rate')

        before_train_qa = before_train.get('qa_accuracy')
        after_train_qa = after_train.get('qa_accuracy')
        before_val_qa = before_val.get('qa_accuracy')
        after_val_qa = after_val.get('qa_accuracy')

        has_comparison = (before_train_em is not None and after_train_em is not None) or \
                         (before_train_qa is not None and after_train_qa is not None)

        if has_comparison:
            print(f"\n  ┌─ Comparison " + "─" * 47)
            print("  │")

            # QA comparison
            if before_train_qa is not None and after_train_qa is not None:
                train_qa_imp = after_train_qa - before_train_qa
                print(f"  │  QA (Train):      {before_train_qa:5.1f}% → {after_train_qa:5.1f}% ({train_qa_imp:+5.1f}%)")
            if before_val_qa is not None and after_val_qa is not None:
                val_qa_imp = after_val_qa - before_val_qa
                print(f"  │  QA (Validation): {before_val_qa:5.1f}% → {after_val_qa:5.1f}% ({val_qa_imp:+5.1f}%)")

            # Judgment comparison
            if before_train_em is not None and after_train_em is not None:
                train_em_imp = after_train_em - before_train_em
                print(f"  │  Judgment (Train):      {before_train_em:5.1f}% → {after_train_em:5.1f}% ({train_em_imp:+5.1f}%)")
            if before_val_em is not None and after_val_em is not None:
                val_em_imp = after_val_em - before_val_em
                print(f"  │  Judgment (Validation): {before_val_em:5.1f}% → {after_val_em:5.1f}% ({val_em_imp:+5.1f}%)")

            print("  │")
            print("  └" + "─" * 60)

    print("\n" + "=" * 70)


def save_phase_results(phase_output: Path, results: dict, filename: str = "eval_results.json"):
    """Save phase results to JSON file."""
    results_path = phase_output / filename

    # Convert any non-serializable types
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    serializable = make_serializable(results)

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"Results saved to: {results_path}")
    return results_path


def load_phase_results(phase_output: Path, filename: str = "eval_results.json") -> dict:
    """Load phase results from JSON file."""
    results_path = phase_output / filename
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}


def is_eval_results_valid(results: dict, phase: int) -> bool:
    """
    Check if evaluation results are valid for a given phase.

    Returns True only if the results contain the required metrics.
    """
    if not results:
        return False

    if phase == 1:
        # Phase 1 needs judgment metrics (exact_match_rate)
        after_train = results.get('after_train', {})
        after_val = results.get('after_val', {})
        return (after_train.get('exact_match_rate') is not None or
                after_val.get('exact_match_rate') is not None)

    elif phase == 2:
        # Phase 2 needs QA metrics (qa_accuracy)
        after_train = results.get('after_train', {})
        after_val = results.get('after_val', {})
        return (after_train.get('qa_accuracy') is not None or
                after_val.get('qa_accuracy') is not None)

    elif phase == 3:
        # Phase 3 needs both judgment and QA metrics
        judgment_train = results.get('judgment_train', {})
        judgment_val = results.get('judgment_val', {})
        qa_train = results.get('qa_train', {})
        qa_val = results.get('qa_val', {})

        has_judgment = (judgment_train.get('exact_match_rate') is not None or
                        judgment_val.get('exact_match_rate') is not None)
        has_qa = (qa_train.get('qa_accuracy') is not None or
                  qa_val.get('qa_accuracy') is not None)
        return has_judgment and has_qa

    return False


# ============== ModelManager: Unified Model Lifecycle ==============

class ModelManager:
    """
    Centralized model lifecycle management for multi-phase training.

    Handles model loading, saving, and provides access to raw model for training.
    Avoids repeated load/unload cycles between phases.
    """

    def __init__(self, ddp: bool, local_rank: int):
        self.ddp = ddp
        self.local_rank = local_rank
        self.model = None
        self.tokenizer = None
        self.raw_model = None  # Unwrapped model (without DDP wrapper)
        self.current_path = None

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None

    def load(self, model_path: str):
        """
        Load model for training.

        If the same model is already loaded, skip reloading.
        """
        model_path = str(model_path)

        if self.model is not None and self.current_path == model_path:
            if is_main_process():
                print(f"Model already loaded from {model_path}, skipping reload")
            return

        # If a different model is loaded, we need to handle the transition
        # For simplicity, we allow loading a new model (overwriting the old one)
        # The old model's weights are preserved if it was saved

        if is_main_process():
            print(f"\n[ModelManager] Loading model from: {model_path}")

        self.model, self.tokenizer = setup_model_for_training(
            model_path,
            use_lora=False,  # Always full fine-tuning
            ddp=self.ddp,
            local_rank=self.local_rank,
        )
        self.current_path = model_path

        # Get raw model (unwrapped from DDP)
        if hasattr(self.model, 'module'):
            self.raw_model = self.model.module
        else:
            self.raw_model = self.model

        if is_main_process():
            print(f"[ModelManager] Model loaded successfully")

    def save(self, path: str):
        """
        Save current model to path.

        Updates current_path to the new save location.
        """
        path = str(path)

        if self.raw_model is None:
            raise ValueError("No model loaded to save")

        if is_main_process():
            print(f"\n[ModelManager] Saving model to: {path}")
            Path(path).mkdir(parents=True, exist_ok=True)
            self.raw_model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"[ModelManager] Model saved successfully")

        self.current_path = path

        # Synchronize
        if dist.is_initialized():
            dist.barrier()

    def update_raw_model(self, raw_model):
        """
        Update raw_model reference after training.

        Some trainers may modify the model in place, call this to ensure
        raw_model reference is up to date.
        """
        self.raw_model = raw_model
        if hasattr(self.model, 'module'):
            # DDP wrapper - the module should already be updated
            pass
        else:
            self.model = raw_model

    def cleanup(self):
        """
        Release model and free GPU memory.

        Call this before running inference that needs GPU memory.
        """
        if is_main_process():
            print("\n[ModelManager] Cleaning up model from GPU memory...")

        if self.model is not None:
            del self.model
        if self.raw_model is not None:
            del self.raw_model
        if self.tokenizer is not None:
            del self.tokenizer

        self.model = None
        self.raw_model = None
        self.tokenizer = None
        self.current_path = None

        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if is_main_process():
            print("[ModelManager] Cleanup complete")

    def batch_inference(
        self,
        samples: list,
        num_trials: int = 3,
        prompt_formatter=None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        batch_size: int = 4,
    ) -> list:
        """
        Run batch inference using the already-loaded model.

        Only main process runs inference; results are broadcast to all processes.

        Args:
            samples: List of sample dicts
            num_trials: Number of responses per sample
            prompt_formatter: Function to format sample into prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            batch_size: Batch size for generation

        Returns:
            List of samples with 'responses' field added
        """
        if self.raw_model is None:
            raise ValueError("No model loaded for inference")

        if prompt_formatter is None:
            prompt_formatter = lambda s: f"Question: {s['question']}\nAnswer:"

        results = []

        if is_main_process():
            print(f"\n[ModelManager] Running inference on {len(samples)} samples, {num_trials} trials each...")

            self.raw_model.eval()

            # Prepare all prompts (sample x trials)
            all_prompts = []
            prompt_indices = []  # (sample_idx, trial_idx)
            for idx, sample in enumerate(samples):
                prompt = prompt_formatter(sample)
                for trial in range(num_trials):
                    all_prompts.append(prompt)
                    prompt_indices.append((idx, trial))

            # Initialize response storage
            sample_responses = [[] for _ in samples]

            # Process in batches
            with torch.no_grad():
                for batch_start in tqdm(range(0, len(all_prompts), batch_size), desc="Inference"):
                    batch_prompts = all_prompts[batch_start:batch_start + batch_size]
                    batch_indices = prompt_indices[batch_start:batch_start + batch_size]

                    # Tokenize
                    inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    ).to(self.raw_model.device)

                    # Generate
                    outputs = self.raw_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if temperature > 0 else None,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )

                    # Decode and store responses
                    for i, (sample_idx, trial_idx) in enumerate(batch_indices):
                        # Get only the generated part
                        input_len = inputs.input_ids[i].shape[0]
                        generated_ids = outputs[i][input_len:]
                        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                        sample_responses[sample_idx].append(response)

            # Build result samples
            for idx, sample in enumerate(samples):
                result_sample = sample.copy()
                result_sample['responses'] = sample_responses[idx]
                results.append(result_sample)

            print(f"[ModelManager] Inference complete, generated {len(all_prompts)} responses")

        # Broadcast results to all processes
        if dist.is_initialized():
            # Serialize results on main process
            if is_main_process():
                results_json = json.dumps(results)
            else:
                results_json = None

            # Broadcast length first
            if is_main_process():
                length_tensor = torch.tensor([len(results_json)], dtype=torch.long, device=f"cuda:{self.local_rank}")
            else:
                length_tensor = torch.tensor([0], dtype=torch.long, device=f"cuda:{self.local_rank}")
            dist.broadcast(length_tensor, src=0)

            # Broadcast data
            if is_main_process():
                data_tensor = torch.tensor(
                    [ord(c) for c in results_json],
                    dtype=torch.uint8,
                    device=f"cuda:{self.local_rank}"
                )
            else:
                data_tensor = torch.zeros(
                    length_tensor.item(),
                    dtype=torch.uint8,
                    device=f"cuda:{self.local_rank}"
                )
            dist.broadcast(data_tensor, src=0)

            # Deserialize on other processes
            if not is_main_process():
                results_json = ''.join([chr(c) for c in data_tensor.cpu().tolist()])
                results = json.loads(results_json)

            dist.barrier()

        return results


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
    actual_abilities = []
    qa_correct_count = 0

    for i, sample in enumerate(samples):
        # Get QA responses for this sample
        start_idx = i * num_trials
        end_idx = start_idx + num_trials
        responses = all_qa_responses[start_idx:end_idx]

        # Determine actual ability
        gold_answers = sample.get("normalized_answers", sample.get("answers", []))
        correct_responses = sum(1 for r in responses if is_correct(r, gold_answers))
        actual_ability = classify_ability(correct_responses, num_trials)
        actual_abilities.append(actual_ability)

        # Track QA accuracy (any correct response)
        if correct_responses > 0:
            qa_correct_count += 1

        # Get predicted ability
        predicted_ability = predicted_abilities[i]

        # Update counts
        pred_dist[predicted_ability] += 1
        actual_dist[actual_ability] += 1

        if predicted_ability == actual_ability:
            correct_count += 1

    total = len(samples)
    exact_match_rate = correct_count / total * 100 if total > 0 else 0
    qa_accuracy = qa_correct_count / total * 100 if total > 0 else 0

    # Compute confusion matrix
    confusion = compute_confusion_matrix(predicted_abilities, actual_abilities)

    print(f"\nResults:")
    print(f"  Exact match rate: {exact_match_rate:.1f}%")
    print(f"  QA accuracy: {qa_accuracy:.1f}%")
    print(f"  Predicted distribution: can={pred_dist['can']}, uncertain={pred_dist['uncertain']}, cannot={pred_dist['cannot']}")
    print(f"  Actual distribution: can={actual_dist['can']}, uncertain={actual_dist['uncertain']}, cannot={actual_dist['cannot']}")

    return {
        "exact_match_rate": exact_match_rate,
        "qa_accuracy": qa_accuracy,
        "qa_correct": qa_correct_count,
        "qa_total": total,
        "pred_can": pred_dist["can"],
        "pred_uncertain": pred_dist["uncertain"],
        "pred_cannot": pred_dist["cannot"],
        "actual_can": actual_dist["can"],
        "actual_uncertain": actual_dist["uncertain"],
        "actual_cannot": actual_dist["cannot"],
        "confusion": confusion,
    }


# ============== Utility Functions ==============

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
        "training_mode": "full_fine_tuning",
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


def is_step_completed(phase: int, step: str, pipeline: MultiPhasePipeline) -> bool:
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


def is_phase_completed(phase: int, pipeline: MultiPhasePipeline) -> bool:
    """
    Check if a phase has been fully completed by checking eval results.

    Returns True only if the phase has valid evaluation results saved.
    This is more reliable than checking model folders, which may be empty or incomplete.
    """
    if phase == 1:
        phase_output = pipeline.get_phase_output_dir("phase1_judgment")
        results = load_phase_results(phase_output, "after_train_results.json")
        # Check if we have valid after-training judgment results
        after_train = results.get('after_train', {})
        after_val = results.get('after_val', {})
        return (after_train.get('exact_match_rate') is not None or
                after_val.get('exact_match_rate') is not None)

    elif phase == 2:
        phase_output = pipeline.get_phase_output_dir("phase2_knowledge")
        results = load_phase_results(phase_output, "after_train_results.json")
        # Check if we have valid after-training QA results
        after_train = results.get('after_train', {})
        after_val = results.get('after_val', {})
        return (after_train.get('qa_accuracy') is not None or
                after_val.get('qa_accuracy') is not None)

    elif phase == 3:
        phase_output = pipeline.get_phase_output_dir("phase3_judgment")
        results = load_phase_results(phase_output, "eval_results.json")
        # Check if we have valid final judgment results
        judgment_train = results.get('judgment_train', {})
        judgment_val = results.get('judgment_val', {})
        return (judgment_train.get('exact_match_rate') is not None or
                judgment_val.get('exact_match_rate') is not None)

    return False


# ============== Phase 1: Data Preparation (Inference) ==============

def run_phase1_data_prep(args, pipeline: MultiPhasePipeline) -> tuple:
    """
    Phase 1 data preparation: collect responses and build training data.

    Uses create_inference for multi-GPU inference (separate from training model).
    Only runs on rank 0, other ranks wait at barrier.

    Returns:
        (samples, training_data) tuple
    """
    project_root = Path(__file__).resolve().parent.parent
    phase_output = pipeline.get_phase_output_dir("phase1_judgment")

    # Step 1.1: Collect responses
    responses_path = phase_output / "responses.jsonl"
    if not args.force and is_step_completed(1, "1.1_responses", pipeline):
        if is_main_process():
            print(f"\n[Step 1.1] Responses already collected, loading from {responses_path}")
    else:
        if is_main_process():
            print(f"\n[Step 1.1] Collecting responses from train split...")

            raw_samples = load_triviaqa(split="train", num_samples=args.num_samples)

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

            for sample in samples:
                gold_answers = sample.get("normalized_answers", sample.get("answers", []))
                evaluation = evaluate_responses(sample["responses"], gold_answers)
                sample["evaluation"] = evaluation
                sample["ability"] = classify_ability(evaluation["correct_count"], evaluation["total"])

            if hasattr(inference, 'shutdown'):
                inference.shutdown()
            del inference
            torch.cuda.empty_cache()

            save_to_jsonl(samples, str(responses_path))
            print(f"Saved {len(samples)} samples to {responses_path}")

    # Synchronize
    if dist.is_initialized():
        dist.barrier()

    # All processes load
    samples = load_from_jsonl(str(responses_path))

    # Step 1.2: Build training data
    training_data_path = phase_output / "training_data.jsonl"
    if not args.force and is_step_completed(1, "1.2_training_data", pipeline):
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

    return samples, training_data


def is_baseline_results_valid(results: dict, phase: int) -> bool:
    """Check if baseline results are valid for a given phase."""
    if not results:
        return False

    if phase == 1:
        # Phase 1 baseline needs judgment metrics
        before_train = results.get('before_train', {})
        before_val = results.get('before_val', {})
        return (before_train.get('exact_match_rate') is not None or
                before_val.get('exact_match_rate') is not None)

    elif phase == 2:
        # Phase 2 baseline needs QA metrics
        before_train = results.get('before_train', {})
        before_val = results.get('before_val', {})
        return (before_train.get('qa_accuracy') is not None or
                before_val.get('qa_accuracy') is not None)

    return False


def run_phase1_baseline_eval(args, pipeline: MultiPhasePipeline) -> dict:
    """
    Phase 1 baseline evaluation (before training).

    Uses subprocess for evaluation to avoid GPU memory conflicts.
    Results are cached and can be loaded if valid.
    Re-runs evaluation if cached results are invalid (regardless of --force).
    """
    phase_output = pipeline.get_phase_output_dir("phase1_judgment")
    results_file = "baseline_results.json"

    # Check for cached results - validate they are complete
    cached_results = load_phase_results(phase_output, results_file)
    if is_baseline_results_valid(cached_results, phase=1) and not args.force:
        if is_main_process():
            print(f"\n[Phase 1] Loading cached baseline results from {phase_output / results_file}")
        return cached_results

    # Results invalid or force mode - need to re-run evaluation
    if is_main_process() and cached_results and not is_baseline_results_valid(cached_results, phase=1):
        print(f"\n[Phase 1] Cached baseline results are invalid, re-running evaluation...")

    eval_results = {}

    if is_main_process():
        print("\n" + "=" * 60)
        print("Phase 1 Baseline Evaluation (Before Training)")
        print("=" * 60)

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

        print("\n[Step 1.3c] Baseline QA accuracy...")
        val_samples = load_triviaqa(split="validation", num_samples=args.test_samples)
        qa_before = test_qa_accuracy(
            args.model, val_samples, args.num_trials,
            args.inference_batch_size, args.num_gpus
        )
        eval_results['before_val'].update(qa_before)
        print(f"  Baseline QA: {qa_before['qa_accuracy']:.1f}%")

        # Save results
        save_phase_results(phase_output, eval_results, results_file)

    if dist.is_initialized():
        dist.barrier()

    return eval_results


def run_phase1_training(args, pipeline: MultiPhasePipeline, model_mgr: ModelManager, training_data: list):
    """
    Phase 1 training: train judgment ability.

    Uses model from ModelManager, updates raw_model reference after training.
    """
    phase_output = pipeline.get_phase_output_dir("phase1_judgment")
    adapter_path = phase_output / "judgment_v1"

    if not args.force and is_step_completed(1, "1.3_train", pipeline):
        if is_main_process():
            print(f"\n[Step 1.4] Judgment model already trained at {adapter_path}")
        return

    if is_main_process():
        print(f"\n[Step 1.4] Training judgment ability with DDP (full fine-tuning)...")

    if args.adaptive:
        from src.ddp_adaptive_trainer import DDPAdaptiveJudgmentTrainer

        trainer = DDPAdaptiveJudgmentTrainer(
            model=model_mgr.model,
            tokenizer=model_mgr.tokenizer,
            learning_rate=args.lr,
            local_rank=model_mgr.local_rank,
        )

        stats = trainer.train_dataset(
            training_data,
            system_prompt=SYSTEM_PROMPT,
            num_epochs=args.epochs,
            skip_correct=True,
        )

        # Update raw_model reference
        model_mgr.update_raw_model(trainer.raw_model)

        if is_main_process():
            print(f"Training stats: {stats['per_epoch'][-1]}")

    # Save model
    model_mgr.save(adapter_path)

    if dist.is_initialized():
        dist.barrier()


def run_phase1_evaluation(args, pipeline: MultiPhasePipeline, model_mgr: ModelManager, samples: list, baseline_results: dict = None) -> dict:
    """
    Phase 1 evaluation after training.

    Uses pre-loaded model from ModelManager.
    Results are cached and can be loaded if valid.
    Re-runs evaluation if cached results are invalid (regardless of --force).
    """
    phase_output = pipeline.get_phase_output_dir("phase1_judgment")
    results_file = "after_train_results.json"

    # Check for cached results - validate they are complete
    cached_results = load_phase_results(phase_output, results_file)
    if is_eval_results_valid(cached_results, phase=1) and not args.force:
        if is_main_process():
            print(f"\n[Phase 1] Loading cached after-training results from {phase_output / results_file}")
            # Merge with baseline and print summary
            all_results = {**(baseline_results or {}), **cached_results}
            print_phase_summary(1, "Initial Judgment Training", all_results, model_mgr.current_path)
        return cached_results

    # Results invalid or force mode - need to re-run evaluation
    if is_main_process() and cached_results and not is_eval_results_valid(cached_results, phase=1):
        print(f"\n[Phase 1] Cached results are invalid, re-running evaluation...")

    eval_results = {}

    if is_main_process() and model_mgr.raw_model is not None:
        print("\n" + "=" * 60)
        print("Phase 1 After-Training Evaluation")
        print("=" * 60)

        train_test_samples = samples[:args.num_samples]
        val_test_samples = load_triviaqa(split="validation", num_samples=args.test_samples)

        print("\n[Step 1.5a] After training JUDGMENT evaluation (train split)...")
        eval_results['after_train'] = evaluate_judgment_with_model(
            model_mgr.raw_model, model_mgr.tokenizer, train_test_samples,
            args.num_trials, args.inference_batch_size
        )

        print("\n[Step 1.5b] After training JUDGMENT evaluation (validation split)...")
        eval_results['after_val'] = evaluate_judgment_with_model(
            model_mgr.raw_model, model_mgr.tokenizer, val_test_samples,
            args.num_trials, args.inference_batch_size
        )

        # Save results
        save_phase_results(phase_output, eval_results, results_file)

        # Merge with baseline and print summary
        all_results = {**(baseline_results or {}), **eval_results}
        print_phase_summary(1, "Initial Judgment Training", all_results, model_mgr.current_path)

    if dist.is_initialized():
        dist.barrier()

    return eval_results


# ============== Phase 2: Knowledge Learning ==============

def run_phase2_data_prep(args, pipeline: MultiPhasePipeline, samples: list) -> list:
    """
    Phase 2 data preparation: build QA dataset.
    """
    phase_output = pipeline.get_phase_output_dir("phase2_knowledge")

    qa_data_path = phase_output / "qa_training_data.jsonl"
    if not args.force and is_step_completed(2, "2.1_qa_data", pipeline):
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

    return qa_data


def run_phase2_baseline_eval(args, pipeline: MultiPhasePipeline, model_path: str, samples: list) -> dict:
    """
    Phase 2 baseline evaluation (QA accuracy before training).

    Results are cached and can be loaded if valid.
    Re-runs evaluation if cached results are invalid (regardless of --force).
    """
    phase_output = pipeline.get_phase_output_dir("phase2_knowledge")
    results_file = "baseline_results.json"

    # Check for cached results - validate they are complete
    cached_results = load_phase_results(phase_output, results_file)
    if is_baseline_results_valid(cached_results, phase=2) and not args.force:
        if is_main_process():
            print(f"\n[Phase 2] Loading cached baseline results from {phase_output / results_file}")
        return cached_results

    # Results invalid or force mode - need to re-run evaluation
    if is_main_process() and cached_results and not is_baseline_results_valid(cached_results, phase=2):
        print(f"\n[Phase 2] Cached baseline results are invalid, re-running evaluation...")

    eval_results = {}

    if is_main_process():
        print("\n" + "=" * 60)
        print("Phase 2 Baseline Evaluation (Before Training)")
        print("=" * 60)

        train_test_samples = samples[:args.test_samples]
        val_test_samples = load_triviaqa(split="validation", num_samples=args.test_samples)

        print("\n[Step 2.2a] Baseline QA accuracy (train split)...")
        before_train = test_qa_accuracy(
            model_path, train_test_samples, args.num_trials,
            args.inference_batch_size, args.num_gpus
        )
        print(f"  Accuracy: {before_train['qa_accuracy']:.1f}%")
        eval_results['before_train'] = before_train

        print("\n[Step 2.2b] Baseline QA accuracy (validation split)...")
        before_val = test_qa_accuracy(
            model_path, val_test_samples, args.num_trials,
            args.inference_batch_size, args.num_gpus
        )
        print(f"  Accuracy: {before_val['qa_accuracy']:.1f}%")
        eval_results['before_val'] = before_val

        # Save results
        save_phase_results(phase_output, eval_results, results_file)

    if dist.is_initialized():
        dist.barrier()

    return eval_results


def run_phase2_training(args, pipeline: MultiPhasePipeline, model_mgr: ModelManager, qa_data: list):
    """
    Phase 2 training: knowledge learning.
    """
    phase_output = pipeline.get_phase_output_dir("phase2_knowledge")
    adapter_path = phase_output / "knowledge"

    if not args.force and is_step_completed(2, "2.2_train", pipeline):
        if is_main_process():
            print(f"\n[Step 2.3] Knowledge model already trained at {adapter_path}")
        return

    if is_main_process():
        print(f"\n[Step 2.3] Training knowledge model with DDP (full fine-tuning)...")

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
        })

    if args.adaptive:
        from src.ddp_adaptive_trainer import DDPAdaptiveKnowledgeTrainer

        trainer = DDPAdaptiveKnowledgeTrainer(
            model=model_mgr.model,
            tokenizer=model_mgr.tokenizer,
            learning_rate=args.lr,
            local_rank=model_mgr.local_rank,
        )

        stats = trainer.train_dataset(
            adaptive_samples,
            num_epochs=args.knowledge_epochs,
            skip_correct=True,
        )

        # Update raw_model reference
        model_mgr.update_raw_model(trainer.raw_model)

        if is_main_process():
            print(f"Training stats: {stats['per_epoch'][-1]}")

    # Save model
    model_mgr.save(adapter_path)

    if dist.is_initialized():
        dist.barrier()


def run_phase2_evaluation(args, pipeline: MultiPhasePipeline, model_mgr: ModelManager, samples: list, baseline_results: dict = None) -> dict:
    """
    Phase 2 evaluation after training.

    Results are cached and can be loaded if valid.
    Re-runs evaluation if cached results are invalid (regardless of --force).
    """
    phase_output = pipeline.get_phase_output_dir("phase2_knowledge")
    results_file = "after_train_results.json"

    # Check for cached results - validate they are complete
    cached_results = load_phase_results(phase_output, results_file)
    if is_eval_results_valid(cached_results, phase=2) and not args.force:
        if is_main_process():
            print(f"\n[Phase 2] Loading cached after-training results from {phase_output / results_file}")
            # Merge with baseline and print summary
            all_results = {**(baseline_results or {}), **cached_results}
            print_phase_summary(2, "Knowledge Learning", all_results, model_mgr.current_path)
        return cached_results

    # Results invalid or force mode - need to re-run evaluation
    if is_main_process() and cached_results and not is_eval_results_valid(cached_results, phase=2):
        print(f"\n[Phase 2] Cached results are invalid, re-running evaluation...")

    eval_results = {}

    # IMPORTANT: Set model to eval mode on ALL ranks before evaluation
    # This prevents collective operation mismatches when only rank 0 does inference
    if model_mgr.raw_model is not None:
        model_mgr.raw_model.eval()

    # Sync all ranks before single-rank evaluation begins
    if dist.is_initialized():
        dist.barrier()

    if is_main_process() and model_mgr.raw_model is not None:
        print("\n" + "=" * 60)
        print("Phase 2 After-Training Evaluation")
        print("=" * 60)

        train_test_samples = samples[:args.test_samples]
        val_test_samples = load_triviaqa(split="validation", num_samples=args.test_samples)

        print("\n[Step 2.4a] After training QA accuracy (train split)...")
        after_train = test_qa_accuracy(
            str(model_mgr.current_path),
            train_test_samples, args.num_trials,
            args.inference_batch_size, args.num_gpus,
            model=model_mgr.raw_model, tokenizer=model_mgr.tokenizer
        )
        print(f"  Accuracy: {after_train['qa_accuracy']:.1f}%")
        eval_results['after_train'] = after_train

        print("\n[Step 2.4b] After training QA accuracy (validation split)...")
        after_val = test_qa_accuracy(
            str(model_mgr.current_path),
            val_test_samples, args.num_trials,
            args.inference_batch_size, args.num_gpus,
            model=model_mgr.raw_model, tokenizer=model_mgr.tokenizer
        )
        print(f"  Accuracy: {after_val['qa_accuracy']:.1f}%")
        eval_results['after_val'] = after_val

        # Save results
        save_phase_results(phase_output, eval_results, results_file)

        # Merge with baseline and print summary
        all_results = {**(baseline_results or {}), **eval_results}
        print_phase_summary(2, "Knowledge Learning", all_results, model_mgr.current_path)

    if dist.is_initialized():
        dist.barrier()

    return eval_results


# ============== Phase 3: Update Judgment ==============

def run_phase3_data_prep(
    args,
    pipeline: MultiPhasePipeline,
    knowledge_model_path: str,
    original_samples: list,
    model_mgr: ModelManager = None,
) -> tuple:
    """
    Phase 3 data preparation: re-collect responses with knowledge model.

    If model_mgr is provided and has the knowledge model loaded, uses it directly
    for inference. Otherwise falls back to create_inference (requires cleanup first).
    """
    project_root = Path(__file__).resolve().parent.parent
    phase1_output = pipeline.get_phase_output_dir("phase1_judgment")
    phase3_output = pipeline.get_phase_output_dir("phase3_judgment")

    # Step 3.1: Re-collect responses
    responses_path = phase3_output / "responses_post_knowledge.jsonl"
    if not args.force and is_step_completed(3, "3.1_responses", pipeline):
        if is_main_process():
            print(f"\n[Step 3.1] Responses already collected, loading from {responses_path}")
    else:
        if is_main_process():
            print(f"\n[Step 3.1] Re-collecting responses with knowledge model...")

        # Use model_mgr if available (avoids reload)
        if model_mgr is not None and model_mgr.is_loaded():
            if is_main_process():
                print(f"Using already-loaded model from ModelManager")

            new_samples = model_mgr.batch_inference(
                samples=original_samples,
                num_trials=args.num_trials,
                prompt_formatter=lambda s: f"Question: {s['question']}\nAnswer:",
                batch_size=args.inference_batch_size,
                temperature=1.0,
            )

            # Evaluate responses (all processes have the results now)
            for sample in new_samples:
                gold_answers = sample.get("normalized_answers", sample.get("answers", []))
                evaluation = evaluate_responses(sample["responses"], gold_answers)
                sample["evaluation"] = evaluation
                sample["ability"] = classify_ability(evaluation["correct_count"], evaluation["total"])

            if is_main_process():
                save_to_jsonl(new_samples, str(responses_path))

                # Show distribution change
                new_dist = {}
                for s in new_samples:
                    ability = s.get("ability", "unknown")
                    new_dist[ability] = new_dist.get(ability, 0) + 1
                print(f"New ability distribution: {new_dist}")
        else:
            # Fallback: use create_inference (requires model_mgr.cleanup() before calling)
            if is_main_process():
                print(f"Using create_inference (model not pre-loaded)")

                inference = create_inference(
                    model_name=knowledge_model_path,
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

    # Synchronize
    if dist.is_initialized():
        dist.barrier()

    # All processes load
    new_samples = load_from_jsonl(str(responses_path))

    # Step 3.2: Build new training data
    training_data_path = phase3_output / "training_data_v2.jsonl"
    if not args.force and is_step_completed(3, "3.2_training_data", pipeline):
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

    return new_samples, training_data


def run_phase3_training(args, pipeline: MultiPhasePipeline, model_mgr: ModelManager, training_data: list):
    """Phase 3 training: update judgment ability."""
    phase_output = pipeline.get_phase_output_dir("phase3_judgment")
    adapter_path = phase_output / "judgment_v2"

    if not args.force and is_step_completed(3, "3.3_train", pipeline):
        if is_main_process():
            print(f"\n[Step 3.3] Judgment v2 model already trained at {adapter_path}")
        return

    if is_main_process():
        print(f"\n[Step 3.3] Training judgment v2 with DDP (full fine-tuning)...")

    if args.adaptive:
        from src.ddp_adaptive_trainer import DDPAdaptiveJudgmentTrainer

        trainer = DDPAdaptiveJudgmentTrainer(
            model=model_mgr.model,
            tokenizer=model_mgr.tokenizer,
            learning_rate=args.lr,
            local_rank=model_mgr.local_rank,
        )

        stats = trainer.train_dataset(
            training_data,
            system_prompt=SYSTEM_PROMPT,
            num_epochs=args.epochs,
            skip_correct=True,
        )

        model_mgr.update_raw_model(trainer.raw_model)

        if is_main_process():
            print(f"Training stats: {stats['per_epoch'][-1]}")

    # Save model
    model_mgr.save(adapter_path)

    if dist.is_initialized():
        dist.barrier()


def run_phase3_evaluation(args, pipeline: MultiPhasePipeline, model_mgr: ModelManager, original_samples: list) -> dict:
    """
    Phase 3 final evaluation.

    Results are cached and can be loaded if valid.
    Re-runs evaluation if cached results are invalid (regardless of --force).
    Evaluates both QA and judgment ability on train/val splits.
    """
    phase_output = pipeline.get_phase_output_dir("phase3_judgment")
    results_file = "eval_results.json"

    # Check for cached results - validate they are complete
    cached_results = load_phase_results(phase_output, results_file)
    if is_eval_results_valid(cached_results, phase=3) and not args.force:
        if is_main_process():
            print(f"\n[Phase 3] Loading cached evaluation results from {phase_output / results_file}")
            # Reformat for print_phase_summary
            summary_results = {
                'after_train': {
                    **cached_results.get('judgment_train', {}),
                    **cached_results.get('qa_train', {}),
                },
                'after_val': {
                    **cached_results.get('judgment_val', {}),
                    **cached_results.get('qa_val', {}),
                },
            }
            print_phase_summary(3, "Update Judgment", summary_results, str(phase_output / "judgment_v2"))
        return cached_results

    # Results invalid or force mode - need to re-run evaluation
    if is_main_process() and cached_results and not is_eval_results_valid(cached_results, phase=3):
        print(f"\n[Phase 3] Cached results are invalid, re-running evaluation...")

    eval_results = {}

    if is_main_process():
        print("\n" + "=" * 60)
        print("Phase 3 Final Evaluation")
        print("=" * 60)

        eval_model = str(phase_output / "judgment_v2")

        # Judgment evaluation with confusion matrix
        print("\n[Step 3.4a] Final JUDGMENT evaluation (train split)...")
        train_test_samples = original_samples[:args.num_samples]
        val_test_samples = load_triviaqa(split="validation", num_samples=args.test_samples)

        # Load model for evaluation if needed
        if model_mgr.raw_model is not None:
            eval_results['judgment_train'] = evaluate_judgment_with_model(
                model_mgr.raw_model, model_mgr.tokenizer, train_test_samples,
                args.num_trials, args.inference_batch_size
            )
        else:
            eval_results['judgment_train'] = run_judgment_evaluation(
                eval_model, "none", "train", args.num_samples,
                args.num_trials, args.inference_batch_size, args.num_gpus
            )

        print("\n[Step 3.4b] Final JUDGMENT evaluation (validation split)...")
        if model_mgr.raw_model is not None:
            eval_results['judgment_val'] = evaluate_judgment_with_model(
                model_mgr.raw_model, model_mgr.tokenizer, val_test_samples,
                args.num_trials, args.inference_batch_size
            )
        else:
            eval_results['judgment_val'] = run_judgment_evaluation(
                eval_model, "none", "validation", args.test_samples,
                args.num_trials, args.inference_batch_size, args.num_gpus
            )

        # QA evaluation
        print("\n[Step 3.4c] Final QA accuracy (train split)...")
        train_samples_for_qa = original_samples[:args.test_samples]
        qa_train = test_qa_accuracy(
            eval_model, train_samples_for_qa,
            args.num_trials, args.inference_batch_size, args.num_gpus,
            model=model_mgr.raw_model, tokenizer=model_mgr.tokenizer if model_mgr.raw_model else None
        )
        eval_results['qa_train'] = qa_train
        print(f"  QA accuracy (train): {qa_train['qa_accuracy']:.1f}%")

        print("\n[Step 3.4d] Final QA accuracy (validation split)...")
        val_samples_for_qa = load_triviaqa(split="validation", num_samples=args.test_samples)
        qa_val = test_qa_accuracy(
            eval_model, val_samples_for_qa,
            args.num_trials, args.inference_batch_size, args.num_gpus,
            model=model_mgr.raw_model, tokenizer=model_mgr.tokenizer if model_mgr.raw_model else None
        )
        eval_results['qa_val'] = qa_val
        print(f"  QA accuracy (val): {qa_val['qa_accuracy']:.1f}%")

        # Save results
        save_phase_results(phase_output, eval_results, results_file)

        # Reformat for print_phase_summary
        summary_results = {
            'after_train': {
                **eval_results.get('judgment_train', {}),
                **eval_results.get('qa_train', {}),
            },
            'after_val': {
                **eval_results.get('judgment_val', {}),
                **eval_results.get('qa_val', {}),
            },
        }
        print_phase_summary(3, "Update Judgment", summary_results, eval_model)

    if dist.is_initialized():
        dist.barrier()

    return eval_results


# ============== Main Entry Point ==============

def main():
    parser = argparse.ArgumentParser(description="Multi-phase metacognition training with DDP")

    # Model params
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--dataset", type=str, default="triviaqa")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name (auto-generated if not provided)")

    # Phase control
    parser.add_argument("--phase", type=int, default=None,
                        help="Run specific phase only (1, 2, or 3)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed phase")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if step already completed")

    # Data params
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=100)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--inference_batch_size", type=int, default=16)

    # Training params
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--knowledge_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
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

    # Create experiment directory
    experiment_dir = output_base / experiment_name
    if is_main_process():
        experiment_dir.mkdir(parents=True, exist_ok=True)
        save_config_log(experiment_dir, args, experiment_name)

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
        print(f"Multi-phase Pipeline (DDP) - Unified Model Management")
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

    # ============== Unified Model Management ==============
    # Create ModelManager but don't load yet
    model_mgr = ModelManager(args.ddp, local_rank)

    # Determine starting model path based on first phase
    phase1_model_path = pipeline.get_phase_output_dir("phase1_judgment") / "judgment_v1"
    phase2_model_path = pipeline.get_phase_output_dir("phase2_knowledge") / "knowledge"

    # Track samples across phases
    samples = None

    for phase in phases_to_run:
        # Check if phase completed
        if not args.force and is_phase_completed(phase, pipeline):
            if is_main_process():
                print(f"\n{'=' * 60}")
                print(f"Phase {phase} already completed, skipping...")
                print(f"(Use --force to re-run)")
                print(f"{'=' * 60}")
            continue

        if phase == 1:
            if is_main_process():
                print("\n" + "=" * 60)
                print("Phase 1: Initial Judgment Training")
                print("=" * 60)

            # Data prep (uses create_inference, separate from training model)
            samples, training_data = run_phase1_data_prep(args, pipeline)

            # Baseline eval (uses subprocess)
            baseline_eval = run_phase1_baseline_eval(args, pipeline)

            # Load model for training
            model_mgr.load(args.model)

            # Training
            run_phase1_training(args, pipeline, model_mgr, training_data)

            # Evaluation (uses pre-loaded model)
            after_eval = run_phase1_evaluation(args, pipeline, model_mgr, samples, baseline_results=baseline_eval)

            # Record phase result
            if is_main_process():
                pipeline.record_phase_result(
                    phase_name="phase1_judgment",
                    status="completed",
                    metrics={**baseline_eval, **after_eval},
                    output_paths={
                        "responses": str(pipeline.get_phase_output_dir("phase1_judgment") / "responses.jsonl"),
                        "training_data": str(pipeline.get_phase_output_dir("phase1_judgment") / "training_data.jsonl"),
                        "judgment_v1": str(phase1_model_path),
                    }
                )
                pipeline.state.current_phase = 1
                pipeline._save_state()
                print("\nPhase 1 completed!")

        elif phase == 2:
            if is_main_process():
                print("\n" + "=" * 60)
                print("Phase 2: Knowledge Learning")
                print("=" * 60)

            # Load samples if not already loaded
            if samples is None:
                responses_path = pipeline.get_phase_output_dir("phase1_judgment") / "responses.jsonl"
                samples = load_from_jsonl(str(responses_path))

            # Data prep
            qa_data = run_phase2_data_prep(args, pipeline, samples)

            # Determine model to load: use Phase 1 output if available
            if phase1_model_path.exists():
                phase2_base_model = str(phase1_model_path)
            else:
                phase2_base_model = args.model

            # Baseline eval (uses subprocess, model not loaded yet)
            baseline_eval = run_phase2_baseline_eval(args, pipeline, phase2_base_model, samples)

            # Load model for training (or continue using if already loaded from same path)
            if not model_mgr.is_loaded() or model_mgr.current_path != phase2_base_model:
                model_mgr.load(phase2_base_model)

            # Training
            run_phase2_training(args, pipeline, model_mgr, qa_data)

            # Evaluation
            after_eval = run_phase2_evaluation(args, pipeline, model_mgr, samples, baseline_results=baseline_eval)

            # Record phase result
            if is_main_process():
                pipeline.record_phase_result(
                    phase_name="phase2_knowledge",
                    status="completed",
                    metrics={**baseline_eval, **after_eval},
                    output_paths={
                        "qa_data": str(pipeline.get_phase_output_dir("phase2_knowledge") / "qa_training_data.jsonl"),
                        "knowledge": str(phase2_model_path),
                    }
                )
                pipeline.state.current_phase = 2
                pipeline._save_state()
                print("\nPhase 2 completed!")

        elif phase == 3:
            if is_main_process():
                print("\n" + "=" * 60)
                print("Phase 3: Update Judgment with Knowledge")
                print("=" * 60)

            # Load original samples if not available
            if samples is None:
                responses_path = pipeline.get_phase_output_dir("phase1_judgment") / "responses.jsonl"
                samples = load_from_jsonl(str(responses_path))
            original_samples = samples[:args.num_samples]

            # Determine knowledge model path
            if phase2_model_path.exists():
                knowledge_model_path = str(phase2_model_path)
            else:
                knowledge_model_path = args.model
                if is_main_process():
                    print(f"Warning: Knowledge model not found, using original: {args.model}")

            # Ensure knowledge model is loaded
            # If continuing from Phase 2, model_mgr already has it loaded
            # If running Phase 3 standalone, need to load it
            if not model_mgr.is_loaded() or model_mgr.current_path != knowledge_model_path:
                model_mgr.load(knowledge_model_path)

            # Data prep - uses model_mgr for inference (no reload needed)
            new_samples, training_data = run_phase3_data_prep(
                args, pipeline, knowledge_model_path, original_samples, model_mgr=model_mgr
            )

            # Training
            run_phase3_training(args, pipeline, model_mgr, training_data)

            # Evaluation
            eval_results = run_phase3_evaluation(args, pipeline, model_mgr, original_samples)

            # Record phase result
            if is_main_process():
                pipeline.record_phase_result(
                    phase_name="phase3_judgment",
                    status="completed",
                    metrics=eval_results,
                    output_paths={
                        "responses": str(pipeline.get_phase_output_dir("phase3_judgment") / "responses_post_knowledge.jsonl"),
                        "training_data": str(pipeline.get_phase_output_dir("phase3_judgment") / "training_data_v2.jsonl"),
                        "judgment_v2": str(pipeline.get_phase_output_dir("phase3_judgment") / "judgment_v2"),
                    }
                )
                pipeline.state.current_phase = 3
                pipeline._save_state()
                print("\nPhase 3 completed!")

        # Synchronize between phases
        if dist.is_initialized():
            dist.barrier()

    # Final cleanup
    model_mgr.cleanup()

    # Final summary
    if is_main_process():
        print("\n" + "=" * 60)
        print("Pipeline Summary")
        print("=" * 60)
        pipeline.print_summary()

    cleanup_ddp()


if __name__ == "__main__":
    main()
