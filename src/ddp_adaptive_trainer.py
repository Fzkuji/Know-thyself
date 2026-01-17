"""
DDP Adaptive Trainer - Multi-GPU training with gradient synchronization.

Each GPU processes different samples, gradients are synchronized across GPUs,
then parameters are updated together.

Usage:
    accelerate launch --num_processes=8 script.py --ddp
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Dict, Callable, Optional
from tqdm import tqdm


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    """Get total number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1


class DDPAdaptiveKnowledgeTrainer:
    """
    DDP version of AdaptiveKnowledgeTrainer.

    Each GPU processes 1 sample, gradients are synced, then update together.
    Effectively batch_size = num_gpus.
    """

    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = 1e-4,
        local_rank: int = None,
    ):
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate

        # Get distributed info
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_rank = local_rank
        self.device = f"cuda:{local_rank}"

        # Wrap model with DDP
        if dist.is_initialized():
            self.model = DDP(model, device_ids=[local_rank])
            self.raw_model = model  # Keep reference to unwrapped model for generation
        else:
            self.model = model
            self.raw_model = model

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

    def _tokenize_sample(self, sample: Dict) -> Dict:
        """Tokenize a single sample for training."""
        if "messages" in sample:
            text = self.tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False
            )
        else:
            text = sample.get("text", "")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _test_knowledge(
        self,
        question: str,
        gold_answers: List[str],
    ) -> bool:
        """Test if model can answer the question correctly."""
        from src.evaluator import is_correct

        messages = [
            {"role": "system", "content": "Answer the question directly and concisely."},
            {"role": "user", "content": question}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        self.raw_model.eval()
        with torch.no_grad():
            outputs = self.raw_model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        return is_correct(response, gold_answers)

    def train_dataset(
        self,
        samples: List[Dict],
        num_epochs: int = 2,
        filter_by_ability: List[str] = None,
        skip_correct: bool = True,
    ) -> Dict:
        """
        Train on samples with DDP.

        Each GPU processes different samples in parallel.
        Gradients are synchronized before parameter update.
        """
        # Filter by ability if specified
        if filter_by_ability:
            filtered = [s for s in samples if s.get("original_ability", s.get("ability")) in filter_by_ability]
            if is_main_process():
                print(f"Filtered by ability {filter_by_ability}: {len(filtered)}/{len(samples)} samples")
            samples = filtered

        total_samples = len(samples)
        world_size = get_world_size()
        rank = get_rank()

        if is_main_process():
            print(f"\nDDP Training: {world_size} GPUs, {total_samples} samples")
            print(f"Effective batch size: {world_size} (1 sample per GPU)")

        stats = {
            "total_samples": total_samples,
            "epochs": num_epochs,
            "world_size": world_size,
            "per_epoch": [],
        }

        for epoch in range(num_epochs):
            epoch_stats = {
                "total_in_epoch": total_samples,
                "skipped_correct": 0,
                "trained": 0,
                "correct_after": 0,
                "still_wrong": 0,
            }

            losses = []

            if is_main_process():
                print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Process samples in batches of world_size
            num_batches = (total_samples + world_size - 1) // world_size

            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}", disable=not is_main_process())

            for batch_idx in pbar:
                # Each GPU gets a different sample
                sample_idx = batch_idx * world_size + rank

                if sample_idx < total_samples:
                    sample = samples[sample_idx]
                    question = sample.get("question", "")
                    gold_answers = sample.get("normalized_answers", sample.get("answers", []))

                    # Check if already correct
                    if skip_correct and question and gold_answers and self._test_knowledge(question, gold_answers):
                        needs_training = False
                        is_correct_now = True
                    else:
                        needs_training = True
                        is_correct_now = False
                else:
                    # Padding: this GPU has no sample for this batch
                    needs_training = False
                    is_correct_now = False
                    sample = None

                # CRITICAL: Sync all GPUs after inference test before DDP forward pass
                if dist.is_initialized():
                    dist.barrier()

                # Synchronize: all GPUs need to participate in forward/backward
                # even if they don't have a sample (use dummy forward)
                self.model.train()
                self.optimizer.zero_grad()

                if needs_training and sample is not None:
                    # Real training
                    inputs = self._tokenize_sample(sample)
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    loss.backward()
                    loss_value = loss.item()
                else:
                    # Dummy forward to keep DDP happy (zero gradient)
                    # Create a minimal dummy input
                    dummy_input = self.tokenizer("dummy", return_tensors="pt").to(self.device)
                    dummy_input["labels"] = dummy_input["input_ids"].clone()
                    outputs = self.model(**dummy_input)
                    dummy_loss = outputs.loss * 0  # Zero loss
                    dummy_loss.backward()
                    loss_value = 0.0

                # Gradient clipping and sync (DDP automatically syncs gradients)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters (all GPUs update together)
                self.optimizer.step()

                # Gather statistics
                if sample_idx < total_samples:
                    if needs_training:
                        losses.append(loss_value)
                        epoch_stats["trained"] += 1

                        # Test after training
                        if question and gold_answers and self._test_knowledge(question, gold_answers):
                            epoch_stats["correct_after"] += 1
                        else:
                            epoch_stats["still_wrong"] += 1
                    else:
                        epoch_stats["skipped_correct"] += 1
                        if is_correct_now:
                            epoch_stats["correct_after"] += 1

                # Update progress bar
                if is_main_process():
                    processed = min((batch_idx + 1) * world_size, total_samples)
                    pbar.set_postfix({
                        "skip": epoch_stats["skipped_correct"],
                        "train": epoch_stats["trained"],
                        "processed": processed,
                    })

            # Aggregate statistics across GPUs
            if dist.is_initialized():
                # Sum up stats from all GPUs
                for key in ["skipped_correct", "trained", "correct_after", "still_wrong"]:
                    tensor = torch.tensor([epoch_stats[key]], device=self.device)
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    epoch_stats[key] = tensor.item()

            if losses:
                epoch_stats["avg_loss"] = sum(losses) / len(losses)

            stats["per_epoch"].append(epoch_stats)

            # Print epoch summary (main process only)
            if is_main_process():
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Skipped (already correct): {int(epoch_stats['skipped_correct'])}")
                print(f"  Trained: {int(epoch_stats['trained'])}")
                print(f"  Correct after epoch: {int(epoch_stats['correct_after'])}/{total_samples} "
                      f"({epoch_stats['correct_after']/total_samples*100:.1f}%)")
                if epoch_stats.get("avg_loss"):
                    print(f"  Average loss: {epoch_stats['avg_loss']:.4f}")

            # Early stopping if all samples are correct
            if epoch_stats["correct_after"] >= total_samples:
                if is_main_process():
                    print(f"\nAll samples learned! Stopping early at epoch {epoch+1}")
                break

        return stats


class DDPAdaptiveJudgmentTrainer:
    """
    DDP version of AdaptiveJudgmentTrainer.
    """

    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = 1e-4,
        local_rank: int = None,
    ):
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate

        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_rank = local_rank
        self.device = f"cuda:{local_rank}"

        if dist.is_initialized():
            self.model = DDP(model, device_ids=[local_rank])
            self.raw_model = model
        else:
            self.model = model
            self.raw_model = model

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

    def _tokenize_sample(self, sample: Dict) -> Dict:
        """Tokenize a single sample for training."""
        if "messages" in sample:
            text = self.tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False
            )
        else:
            text = sample.get("text", "")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _test_judgment(
        self,
        question: str,
        expected_ability: str,
        system_prompt: str,
    ) -> bool:
        """Test if model predicts the correct ability."""
        import re

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Can you answer this question correctly?\n\nQuestion: {question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        self.raw_model.eval()
        with torch.no_grad():
            outputs = self.raw_model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip().lower()

        # Parse prediction
        match = re.search(r'\\boxed\{(\w+)\}', response)
        if match:
            predicted = match.group(1).lower()
        elif "yes" in response:
            predicted = "yes"
        elif "uncertain" in response:
            predicted = "uncertain"
        else:
            predicted = "no"

        ability_map = {"yes": "can", "uncertain": "uncertain", "no": "cannot"}
        predicted_ability = ability_map.get(predicted, "cannot")

        return predicted_ability == expected_ability

    def train_dataset(
        self,
        samples: List[Dict],
        system_prompt: str,
        num_epochs: int = 2,
        skip_correct: bool = True,
    ) -> Dict:
        """
        Train on judgment samples with DDP.
        """
        total_samples = len(samples)
        world_size = get_world_size()
        rank = get_rank()

        if is_main_process():
            print(f"\nDDP Judgment Training: {world_size} GPUs, {total_samples} samples")

        stats = {
            "total_samples": total_samples,
            "epochs": num_epochs,
            "world_size": world_size,
            "per_epoch": [],
        }

        for epoch in range(num_epochs):
            epoch_stats = {
                "total_in_epoch": total_samples,
                "skipped_correct": 0,
                "trained": 0,
                "correct_after": 0,
                "still_wrong": 0,
                "by_ability": {"can": {"correct": 0, "total": 0},
                               "uncertain": {"correct": 0, "total": 0},
                               "cannot": {"correct": 0, "total": 0}},
            }

            losses = []

            if is_main_process():
                print(f"\nEpoch {epoch+1}/{num_epochs}")

            num_batches = (total_samples + world_size - 1) // world_size
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}", disable=not is_main_process())

            for batch_idx in pbar:
                sample_idx = batch_idx * world_size + rank

                if sample_idx < total_samples:
                    sample = samples[sample_idx]
                    question = sample.get("question", "")
                    ability = sample.get("ability", "")

                    if skip_correct and question and ability and self._test_judgment(question, ability, system_prompt):
                        needs_training = False
                        is_correct_now = True
                    else:
                        needs_training = True
                        is_correct_now = False
                else:
                    needs_training = False
                    is_correct_now = False
                    sample = None
                    ability = ""

                # CRITICAL: Sync all GPUs after inference test before DDP forward pass
                if dist.is_initialized():
                    dist.barrier()

                self.model.train()
                self.optimizer.zero_grad()

                if needs_training and sample is not None:
                    inputs = self._tokenize_sample(sample)
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    loss.backward()
                    loss_value = loss.item()
                else:
                    dummy_input = self.tokenizer("dummy", return_tensors="pt").to(self.device)
                    dummy_input["labels"] = dummy_input["input_ids"].clone()
                    outputs = self.model(**dummy_input)
                    dummy_loss = outputs.loss * 0
                    dummy_loss.backward()
                    loss_value = 0.0

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Update local stats
                if sample_idx < total_samples and ability:
                    epoch_stats["by_ability"][ability]["total"] += 1

                    if needs_training:
                        losses.append(loss_value)
                        epoch_stats["trained"] += 1

                        if question and self._test_judgment(question, ability, system_prompt):
                            epoch_stats["correct_after"] += 1
                            epoch_stats["by_ability"][ability]["correct"] += 1
                        else:
                            epoch_stats["still_wrong"] += 1
                    else:
                        epoch_stats["skipped_correct"] += 1
                        if is_correct_now:
                            epoch_stats["correct_after"] += 1
                            epoch_stats["by_ability"][ability]["correct"] += 1

                if is_main_process():
                    processed = min((batch_idx + 1) * world_size, total_samples)
                    pbar.set_postfix({
                        "skip": epoch_stats["skipped_correct"],
                        "train": epoch_stats["trained"],
                        "processed": processed,
                    })

            # Aggregate stats across GPUs
            if dist.is_initialized():
                for key in ["skipped_correct", "trained", "correct_after", "still_wrong"]:
                    tensor = torch.tensor([epoch_stats[key]], device=self.device)
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    epoch_stats[key] = tensor.item()

                for ability in ["can", "uncertain", "cannot"]:
                    for subkey in ["correct", "total"]:
                        tensor = torch.tensor([epoch_stats["by_ability"][ability][subkey]], device=self.device)
                        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                        epoch_stats["by_ability"][ability][subkey] = tensor.item()

            if losses:
                epoch_stats["avg_loss"] = sum(losses) / len(losses)

            stats["per_epoch"].append(epoch_stats)

            if is_main_process():
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Skipped (already correct): {int(epoch_stats['skipped_correct'])}")
                print(f"  Trained: {int(epoch_stats['trained'])}")
                print(f"  Correct after epoch: {int(epoch_stats['correct_after'])}/{total_samples} "
                      f"({epoch_stats['correct_after']/total_samples*100:.1f}%)")

                print(f"  By ability:")
                for ability in ["can", "uncertain", "cannot"]:
                    ab_stats = epoch_stats["by_ability"][ability]
                    if ab_stats["total"] > 0:
                        acc = ab_stats["correct"] / ab_stats["total"] * 100
                        print(f"    {ability}: {int(ab_stats['correct'])}/{int(ab_stats['total'])} ({acc:.1f}%)")

                if epoch_stats.get("avg_loss"):
                    print(f"  Average loss: {epoch_stats['avg_loss']:.4f}")

            if epoch_stats["correct_after"] >= total_samples:
                if is_main_process():
                    print(f"\nAll samples learned! Stopping early at epoch {epoch+1}")
                break

        return stats


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
