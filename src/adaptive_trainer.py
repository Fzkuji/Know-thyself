"""
Adaptive trainer - train each sample until learned.

Key idea: Instead of training for fixed epochs, train each sample
until the model demonstrates it has learned (can answer correctly
or make correct judgment).
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
                temperature=0.1,  # Low temperature for deterministic check
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

    def train_dataset(
        self,
        samples: List[Dict],
        num_epochs: int = 2,
        progress_callback: Callable = None,
    ) -> Dict:
        """
        Train on all samples adaptively.

        Args:
            samples: List of training samples with 'messages', 'question', 'answers'
            num_epochs: Number of passes through the dataset
            progress_callback: Optional callback for progress updates

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
                "learned": 0,
                "not_learned": 0,
                "total_steps": 0,
            }

            pbar = tqdm(samples, desc=f"Epoch {epoch+1}/{num_epochs}")
            for sample in pbar:
                question = sample.get("question", "")
                gold_answers = sample.get("normalized_answers", sample.get("answers", []))

                if not question or not gold_answers:
                    continue

                result = self.train_sample(sample, question, gold_answers)

                if result["learned"]:
                    epoch_stats["learned"] += 1
                else:
                    epoch_stats["not_learned"] += 1
                epoch_stats["total_steps"] += result["steps"]

                # Update progress bar
                learn_rate = epoch_stats["learned"] / (epoch_stats["learned"] + epoch_stats["not_learned"]) * 100
                pbar.set_postfix({
                    "learned": f"{learn_rate:.1f}%",
                    "steps": epoch_stats["total_steps"]
                })

            stats["per_epoch"].append(epoch_stats)
            print(f"\nEpoch {epoch+1}: Learned {epoch_stats['learned']}/{total_samples} "
                  f"({epoch_stats['learned']/total_samples*100:.1f}%)")

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

    def train_dataset(
        self,
        samples: List[Dict],
        system_prompt: str,
        num_epochs: int = 2,
    ) -> Dict:
        """Train on all judgment samples adaptively."""
        total_samples = len(samples)
        stats = {
            "total_samples": total_samples,
            "epochs": num_epochs,
            "per_epoch": [],
        }

        for epoch in range(num_epochs):
            epoch_stats = {
                "learned": 0,
                "not_learned": 0,
                "total_steps": 0,
            }

            pbar = tqdm(samples, desc=f"Epoch {epoch+1}/{num_epochs}")
            for sample in pbar:
                question = sample.get("question", "")
                ability = sample.get("ability", "")

                if not question or not ability:
                    continue

                result = self.train_sample(sample, question, ability, system_prompt)

                if result["learned"]:
                    epoch_stats["learned"] += 1
                else:
                    epoch_stats["not_learned"] += 1
                epoch_stats["total_steps"] += result["steps"]

                learn_rate = epoch_stats["learned"] / (epoch_stats["learned"] + epoch_stats["not_learned"]) * 100
                pbar.set_postfix({
                    "learned": f"{learn_rate:.1f}%",
                    "steps": epoch_stats["total_steps"]
                })

            stats["per_epoch"].append(epoch_stats)
            print(f"\nEpoch {epoch+1}: Learned {epoch_stats['learned']}/{total_samples} "
                  f"({epoch_stats['learned']/total_samples*100:.1f}%)")

        return stats
