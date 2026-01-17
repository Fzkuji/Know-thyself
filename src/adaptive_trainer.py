"""
Adaptive trainer - train samples with inline testing.

Training flow for each sample in an epoch:
1. Test if model already knows/judges correctly
2. If correct -> skip (don't train)
3. If wrong -> train one step -> test again

This is more efficient than testing all samples before/after each epoch.

For knowledge training, can also filter by ability (only train uncertain/cannot).
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Dict, Callable, Optional
from tqdm import tqdm
import copy


class AdaptiveKnowledgeTrainer:
    """
    Train model on QA pairs until each question can be answered correctly.
    """

    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = 1e-4,
        max_steps_per_sample: int = 10,
        device: str = "auto",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.max_steps_per_sample = max_steps_per_sample

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

    def _tokenize_sample(self, sample: Dict) -> Dict:
        """Tokenize a single sample for training."""
        # Format as chat
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
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _test_knowledge(
        self,
        question: str,
        gold_answers: List[str],
        num_trials: int = 1,
    ) -> bool:
        """Test if model can answer the question correctly."""
        from src.evaluator import is_correct

        # Build prompt for answering
        messages = [
            {"role": "system", "content": "Answer the question directly and concisely."},
            {"role": "user", "content": question}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=1.0,  # Use same temperature as evaluation for consistency
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        self.model.train()
        return is_correct(response, gold_answers)

    def train_sample(
        self,
        sample: Dict,
        question: str,
        gold_answers: List[str],
    ) -> Dict:
        """
        Train on a single sample until learned or max steps reached.

        Returns:
            Dict with training stats
        """
        self.model.train()
        inputs = self._tokenize_sample(sample)

        learned = False
        steps_taken = 0

        for step in range(self.max_steps_per_sample):
            # Forward pass
            outputs = self.model(**inputs)
            loss = outputs.loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            steps_taken += 1

            # Test if learned
            if self._test_knowledge(question, gold_answers):
                learned = True
                break

        return {
            "learned": learned,
            "steps": steps_taken,
            "final_loss": loss.item(),
        }

    def _train_one_step(self, sample: Dict) -> float:
        """Train one gradient step on a sample. Returns loss."""
        self.model.train()
        inputs = self._tokenize_sample(sample)

        outputs = self.model(**inputs)
        loss = outputs.loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train_dataset(
        self,
        samples: List[Dict],
        num_epochs: int = 2,
        progress_callback: Callable = None,
        filter_by_ability: List[str] = None,
        skip_correct: bool = True,
    ) -> Dict:
        """
        Train on samples with epoch-level filtering.

        Args:
            samples: List of training samples with 'messages', 'question', 'answers'
            num_epochs: Number of passes through the dataset
            progress_callback: Optional callback for progress updates
            filter_by_ability: Only train samples with these abilities (e.g., ["cannot", "uncertain"])
            skip_correct: If True, skip samples that model already answers correctly

        Returns:
            Training statistics
        """
        # Filter by ability if specified
        if filter_by_ability:
            filtered = [s for s in samples if s.get("original_ability", s.get("ability")) in filter_by_ability]
            print(f"Filtered by ability {filter_by_ability}: {len(filtered)}/{len(samples)} samples")
            samples = filtered

        total_samples = len(samples)
        stats = {
            "total_samples": total_samples,
            "epochs": num_epochs,
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
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            pbar = tqdm(samples, desc=f"Epoch {epoch+1}")

            for sample in pbar:
                question = sample.get("question", "")
                gold_answers = sample.get("normalized_answers", sample.get("answers", []))

                if not question or not gold_answers:
                    continue

                # Test first, only train if wrong
                if skip_correct and self._test_knowledge(question, gold_answers):
                    epoch_stats["skipped_correct"] += 1
                    epoch_stats["correct_after"] += 1
                else:
                    # Train one step
                    loss = self._train_one_step(sample)
                    losses.append(loss)
                    epoch_stats["trained"] += 1

                    # Test after training
                    if self._test_knowledge(question, gold_answers):
                        epoch_stats["correct_after"] += 1
                    else:
                        epoch_stats["still_wrong"] += 1

                # Update progress bar
                pbar.set_postfix({
                    "skip": epoch_stats["skipped_correct"],
                    "train": epoch_stats["trained"],
                    "acc": f"{epoch_stats['correct_after']/(epoch_stats['skipped_correct']+epoch_stats['trained'])*100:.1f}%" if (epoch_stats['skipped_correct']+epoch_stats['trained']) > 0 else "0%"
                })

            if losses:
                epoch_stats["avg_loss"] = sum(losses) / len(losses)

            stats["per_epoch"].append(epoch_stats)

            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Skipped (already correct): {epoch_stats['skipped_correct']}")
            print(f"  Trained: {epoch_stats['trained']}")
            print(f"  Correct after epoch: {epoch_stats['correct_after']}/{total_samples} "
                  f"({epoch_stats['correct_after']/total_samples*100:.1f}%)")
            if epoch_stats.get("avg_loss"):
                print(f"  Average loss: {epoch_stats['avg_loss']:.4f}")

            # Early stopping if all samples are correct
            if epoch_stats["correct_after"] == total_samples:
                print(f"\nAll samples learned! Stopping early at epoch {epoch+1}")
                break

        return stats


class AdaptiveJudgmentTrainer:
    """
    Train model on judgment task until each sample is correctly judged.
    """

    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = 1e-4,
        max_steps_per_sample: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.max_steps_per_sample = max_steps_per_sample

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
        return {k: v.to(self.model.device) for k, v in inputs.items()}

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

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
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

        self.model.train()

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

        # Map to ability
        ability_map = {"yes": "can", "uncertain": "uncertain", "no": "cannot"}
        predicted_ability = ability_map.get(predicted, "cannot")

        return predicted_ability == expected_ability

    def train_sample(
        self,
        sample: Dict,
        question: str,
        expected_ability: str,
        system_prompt: str,
    ) -> Dict:
        """Train on a single judgment sample until correct or max steps."""
        self.model.train()
        inputs = self._tokenize_sample(sample)

        learned = False
        steps_taken = 0

        for step in range(self.max_steps_per_sample):
            outputs = self.model(**inputs)
            loss = outputs.loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            steps_taken += 1

            if self._test_judgment(question, expected_ability, system_prompt):
                learned = True
                break

        return {
            "learned": learned,
            "steps": steps_taken,
            "final_loss": loss.item(),
        }

    def _train_one_step(self, sample: Dict) -> float:
        """Train one gradient step on a sample. Returns loss."""
        self.model.train()
        inputs = self._tokenize_sample(sample)

        outputs = self.model(**inputs)
        loss = outputs.loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train_dataset(
        self,
        samples: List[Dict],
        system_prompt: str,
        num_epochs: int = 2,
        skip_correct: bool = True,
    ) -> Dict:
        """
        Train on judgment samples with epoch-level filtering.

        Args:
            samples: List of training samples with 'messages', 'question', 'ability'
            system_prompt: System prompt for judgment task
            num_epochs: Number of passes through the dataset
            skip_correct: If True, skip samples that model already judges correctly

        Returns:
            Training statistics
        """
        total_samples = len(samples)
        stats = {
            "total_samples": total_samples,
            "epochs": num_epochs,
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
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            pbar = tqdm(samples, desc=f"Epoch {epoch+1}")

            for sample in pbar:
                question = sample.get("question", "")
                ability = sample.get("ability", "")

                if not question or not ability:
                    continue

                epoch_stats["by_ability"][ability]["total"] += 1

                # Test first, only train if wrong
                if skip_correct and self._test_judgment(question, ability, system_prompt):
                    epoch_stats["skipped_correct"] += 1
                    epoch_stats["correct_after"] += 1
                    epoch_stats["by_ability"][ability]["correct"] += 1
                else:
                    # Train one step
                    loss = self._train_one_step(sample)
                    losses.append(loss)
                    epoch_stats["trained"] += 1

                    # Test after training
                    if self._test_judgment(question, ability, system_prompt):
                        epoch_stats["correct_after"] += 1
                        epoch_stats["by_ability"][ability]["correct"] += 1
                    else:
                        epoch_stats["still_wrong"] += 1

                # Update progress bar
                pbar.set_postfix({
                    "skip": epoch_stats["skipped_correct"],
                    "train": epoch_stats["trained"],
                    "acc": f"{epoch_stats['correct_after']/(epoch_stats['skipped_correct']+epoch_stats['trained'])*100:.1f}%" if (epoch_stats['skipped_correct']+epoch_stats['trained']) > 0 else "0%"
                })

            if losses:
                epoch_stats["avg_loss"] = sum(losses) / len(losses)

            stats["per_epoch"].append(epoch_stats)

            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Skipped (already correct): {epoch_stats['skipped_correct']}")
            print(f"  Trained: {epoch_stats['trained']}")
            print(f"  Correct after epoch: {epoch_stats['correct_after']}/{total_samples} "
                  f"({epoch_stats['correct_after']/total_samples*100:.1f}%)")

            # Per-ability breakdown
            print(f"  By ability:")
            for ability in ["can", "uncertain", "cannot"]:
                ab_stats = epoch_stats["by_ability"][ability]
                if ab_stats["total"] > 0:
                    acc = ab_stats["correct"] / ab_stats["total"] * 100
                    print(f"    {ability}: {ab_stats['correct']}/{ab_stats['total']} ({acc:.1f}%)")

            if epoch_stats.get("avg_loss"):
                print(f"  Average loss: {epoch_stats['avg_loss']:.4f}")

            # Early stopping if all samples are correct
            if epoch_stats["correct_after"] == total_samples:
                print(f"\nAll samples learned! Stopping early at epoch {epoch+1}")
                break

        return stats
