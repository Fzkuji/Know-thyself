"""
Single GPU Adaptive Trainer - Training on single GPU without DDP.

This avoids DDP synchronization issues during adaptive training with generate() calls.

Usage:
    trainer = SingleGPUKnowledgeTrainer(model, tokenizer)
    stats = trainer.train_dataset(samples, num_epochs=2)
"""

import torch
from typing import List, Dict
from tqdm import tqdm


class SingleGPUKnowledgeTrainer:
    """
    Single GPU version of AdaptiveKnowledgeTrainer.

    Processes samples one at a time with adaptive training:
    - Skip samples the model already knows
    - Train until learned
    """

    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = 1e-4,
        device: str = "cuda:0",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.device = device

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

        return is_correct(response, gold_answers)

    def train_dataset(
        self,
        samples: List[Dict],
        num_epochs: int = 2,
        filter_by_ability: List[str] = None,
        skip_correct: bool = True,
    ) -> Dict:
        """
        Train on samples with single GPU.
        """
        # Filter by ability if specified
        if filter_by_ability:
            filtered = [s for s in samples if s.get("original_ability", s.get("ability")) in filter_by_ability]
            print(f"Filtered by ability {filter_by_ability}: {len(filtered)}/{len(samples)} samples")
            samples = filtered

        total_samples = len(samples)

        print(f"\nSingle GPU Training: {total_samples} samples")

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

                # Check if already correct
                if skip_correct and question and gold_answers and self._test_knowledge(question, gold_answers):
                    epoch_stats["skipped_correct"] += 1
                    epoch_stats["correct_after"] += 1
                    pbar.set_postfix({
                        "skip": epoch_stats["skipped_correct"],
                        "train": epoch_stats["trained"],
                    })
                    continue

                # Train on this sample
                self.model.train()
                self.optimizer.zero_grad()

                inputs = self._tokenize_sample(sample)
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()

                losses.append(loss.item())

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_stats["trained"] += 1

                # Test after training
                if question and gold_answers and self._test_knowledge(question, gold_answers):
                    epoch_stats["correct_after"] += 1
                else:
                    epoch_stats["still_wrong"] += 1

                pbar.set_postfix({
                    "skip": epoch_stats["skipped_correct"],
                    "train": epoch_stats["trained"],
                    "loss": loss.item(),
                })

            if losses:
                epoch_stats["avg_loss"] = sum(losses) / len(losses)

            stats["per_epoch"].append(epoch_stats)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Skipped (already correct): {epoch_stats['skipped_correct']}")
            print(f"  Trained: {epoch_stats['trained']}")
            print(f"  Correct after epoch: {epoch_stats['correct_after']}/{total_samples} "
                  f"({epoch_stats['correct_after']/total_samples*100:.1f}%)")
            if epoch_stats.get("avg_loss"):
                print(f"  Average loss: {epoch_stats['avg_loss']:.4f}")

            # Early stopping if all samples are correct
            if epoch_stats["correct_after"] >= total_samples:
                print(f"\nAll samples learned! Stopping early at epoch {epoch+1}")
                break

        return stats


class SingleGPUJudgmentTrainer:
    """
    Single GPU version of AdaptiveJudgmentTrainer.

    Key feature: Real-time label generation during training.
    Instead of using static labels from Phase 1 data collection,
    we generate QA responses on-the-fly and determine the model's
    current ability to answer each question.
    """

    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = 1e-4,
        device: str = "cuda:0",
        num_qa_trials: int = 5,  # Number of QA trials for real-time ability assessment
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.device = device
        self.num_qa_trials = num_qa_trials

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

    def _generate_qa_responses(self, question: str, num_trials: int = None) -> List[str]:
        """
        Generate multiple QA responses for a question to assess current ability.

        Args:
            question: The question to answer
            num_trials: Number of responses to generate (default: self.num_qa_trials)

        Returns:
            List of response strings
        """
        if num_trials is None:
            num_trials = self.num_qa_trials

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

        responses = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_trials):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=1.0,  # Use sampling for QA
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                responses.append(response)

        return responses

    def _compute_realtime_ability(self, question: str, gold_answers: List[str]) -> str:
        """
        Compute the model's current ability to answer a question.

        This generates num_qa_trials responses and classifies ability based on
        how many are correct.

        Args:
            question: The question to assess
            gold_answers: List of acceptable answers

        Returns:
            "can" / "uncertain" / "cannot"
        """
        from src.evaluator import is_correct, classify_ability

        responses = self._generate_qa_responses(question)
        correct_count = sum(1 for r in responses if is_correct(r, gold_answers))
        return classify_ability(correct_count, len(responses))

    def _build_judgment_sample(
        self,
        question: str,
        ability: str,
        system_prompt: str,
    ) -> Dict:
        """
        Build a training sample for judgment prediction.

        Args:
            question: The question
            ability: "can" / "uncertain" / "cannot"
            system_prompt: System prompt for judgment task

        Returns:
            Training sample dict with "messages" field
        """
        ability_to_answer = {
            "can": "Yes",
            "uncertain": "Uncertain",
            "cannot": "No"
        }
        answer = ability_to_answer.get(ability, "No")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Can you answer this question correctly?\n\nQuestion: {question}"},
            {"role": "assistant", "content": f"\\boxed{{{answer}}}"}
        ]

        return {"messages": messages, "ability": ability}

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

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0,
                do_sample=False,  # Greedy decoding for judgment
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
        use_realtime_labels: bool = True,  # NEW: Use real-time ability assessment
    ) -> Dict:
        """
        Train on judgment samples with single GPU.

        Args:
            samples: List of samples with 'question' and 'normalized_answers' fields
            system_prompt: System prompt for judgment task
            num_epochs: Number of training epochs
            skip_correct: Skip samples where judgment is already correct
            use_realtime_labels: If True, compute ability labels in real-time by
                                 generating QA responses. If False, use pre-computed
                                 'ability' field from samples.
        """
        total_samples = len(samples)

        mode_str = "real-time labels" if use_realtime_labels else "static labels"
        print(f"\nSingle GPU Judgment Training: {total_samples} samples ({mode_str})")

        stats = {
            "total_samples": total_samples,
            "epochs": num_epochs,
            "use_realtime_labels": use_realtime_labels,
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
                gold_answers = sample.get("normalized_answers", sample.get("answers", []))

                # Compute ability: either real-time or from pre-computed label
                if use_realtime_labels and question and gold_answers:
                    # Real-time: Generate QA responses and determine current ability
                    ability = self._compute_realtime_ability(question, gold_answers)
                else:
                    # Static: Use pre-computed ability from sample
                    ability = sample.get("ability", "")

                if ability:
                    epoch_stats["by_ability"][ability]["total"] += 1

                # Check if model already predicts correct judgment
                if skip_correct and question and ability and self._test_judgment(question, ability, system_prompt):
                    epoch_stats["skipped_correct"] += 1
                    epoch_stats["correct_after"] += 1
                    if ability:
                        epoch_stats["by_ability"][ability]["correct"] += 1
                    pbar.set_postfix({
                        "skip": epoch_stats["skipped_correct"],
                        "train": epoch_stats["trained"],
                    })
                    continue

                # Build training sample with current ability label
                if use_realtime_labels:
                    train_sample = self._build_judgment_sample(question, ability, system_prompt)
                else:
                    train_sample = sample

                # Train on this sample
                self.model.train()
                self.optimizer.zero_grad()

                inputs = self._tokenize_sample(train_sample)
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()

                losses.append(loss.item())

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_stats["trained"] += 1

                # Test after training
                if question and ability and self._test_judgment(question, ability, system_prompt):
                    epoch_stats["correct_after"] += 1
                    if ability:
                        epoch_stats["by_ability"][ability]["correct"] += 1
                else:
                    epoch_stats["still_wrong"] += 1

                pbar.set_postfix({
                    "skip": epoch_stats["skipped_correct"],
                    "train": epoch_stats["trained"],
                    "loss": loss.item(),
                })

            if losses:
                epoch_stats["avg_loss"] = sum(losses) / len(losses)

            stats["per_epoch"].append(epoch_stats)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Skipped (already correct): {epoch_stats['skipped_correct']}")
            print(f"  Trained: {epoch_stats['trained']}")
            print(f"  Correct after epoch: {epoch_stats['correct_after']}/{total_samples} "
                  f"({epoch_stats['correct_after']/total_samples*100:.1f}%)")

            print(f"  By ability:")
            for ability in ["can", "uncertain", "cannot"]:
                ab_stats = epoch_stats["by_ability"][ability]
                if ab_stats["total"] > 0:
                    acc = ab_stats["correct"] / ab_stats["total"] * 100
                    print(f"    {ability}: {ab_stats['correct']}/{ab_stats['total']} ({acc:.1f}%)")

            if epoch_stats.get("avg_loss"):
                print(f"  Average loss: {epoch_stats['avg_loss']:.4f}")

            # Early stopping if all samples are correct
            if epoch_stats["correct_after"] >= total_samples:
                print(f"\nAll samples learned! Stopping early at epoch {epoch+1}")
                break

        return stats
