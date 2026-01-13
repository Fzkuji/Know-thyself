"""
Model inference module - query model multiple times for each question.
Supports batch inference for better GPU utilization.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
from tqdm import tqdm


class ModelInference:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        inference_batch_size: int = 8,  # Batch size for parallel generation
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.inference_batch_size = inference_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Important for batch generation

    def generate(self, prompt: str, num_samples: int = 1) -> List[str]:
        """
        Generate multiple responses for a single prompt using batch generation.

        Args:
            prompt: Input prompt
            num_samples: Number of responses to generate

        Returns:
            List of generated responses
        """
        # Use num_return_sequences for parallel sampling
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                num_return_sequences=num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        responses = []
        input_len = inputs["input_ids"].shape[1]
        for i in range(num_samples):
            response = self.tokenizer.decode(
                outputs[i][input_len:],
                skip_special_tokens=True
            ).strip()
            responses.append(response)

        return responses

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts in a batch.

        Args:
            prompts: List of prompts

        Returns:
            List of responses (one per prompt)
        """
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        responses = []
        for i, output in enumerate(outputs):
            input_len = (inputs["attention_mask"][i] == 1).sum().item()
            response = self.tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True
            ).strip()
            responses.append(response)

        return responses

    def batch_inference(
        self,
        samples: List[Dict],
        num_trials: int = 5,
        prompt_formatter: callable = None,
        batch_size: int = None,
    ) -> List[Dict]:
        """
        Run inference on multiple samples with batch processing.

        Args:
            samples: List of samples with 'question' field
            num_trials: Number of times to query each question
            prompt_formatter: Function to format question into prompt
            batch_size: Batch size for processing multiple questions

        Returns:
            Samples with added 'responses' field
        """
        if prompt_formatter is None:
            prompt_formatter = lambda q: f"Question: {q}\nAnswer:"

        if batch_size is None:
            batch_size = self.inference_batch_size

        results = []

        # Process in batches of questions
        for batch_start in tqdm(range(0, len(samples), batch_size), desc="Running inference"):
            batch_end = min(batch_start + batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]

            # For each question in batch, generate num_trials responses
            for sample in batch_samples:
                prompt = prompt_formatter(sample["question"])
                responses = self.generate(prompt, num_samples=num_trials)

                result = sample.copy()
                result["responses"] = responses
                results.append(result)

        return results

    def batch_inference_parallel(
        self,
        samples: List[Dict],
        num_trials: int = 5,
        prompt_formatter: callable = None,
        batch_size: int = None,
    ) -> List[Dict]:
        """
        Run inference with maximum parallelism - batch across both questions and trials.

        Args:
            samples: List of samples with 'question' field
            num_trials: Number of times to query each question
            prompt_formatter: Function to format question into prompt
            batch_size: Number of prompts to process at once

        Returns:
            Samples with added 'responses' field
        """
        if prompt_formatter is None:
            prompt_formatter = lambda q: f"Question: {q}\nAnswer:"

        if batch_size is None:
            batch_size = self.inference_batch_size

        # Create all prompts (question x trials)
        all_prompts = []
        prompt_to_sample_idx = []
        for idx, sample in enumerate(samples):
            prompt = prompt_formatter(sample["question"])
            for _ in range(num_trials):
                all_prompts.append(prompt)
                prompt_to_sample_idx.append(idx)

        # Process all prompts in batches
        all_responses = []
        for batch_start in tqdm(range(0, len(all_prompts), batch_size), desc="Running batch inference"):
            batch_end = min(batch_start + batch_size, len(all_prompts))
            batch_prompts = all_prompts[batch_start:batch_end]
            batch_responses = self.generate_batch(batch_prompts)
            all_responses.extend(batch_responses)

        # Group responses by sample
        results = []
        for idx, sample in enumerate(samples):
            sample_responses = []
            for i, resp_idx in enumerate(prompt_to_sample_idx):
                if resp_idx == idx:
                    sample_responses.append(all_responses[i])
                    if len(sample_responses) == num_trials:
                        break

            result = sample.copy()
            result["responses"] = sample_responses
            results.append(result)

        return results
