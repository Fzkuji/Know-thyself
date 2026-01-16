"""
Model inference module - batch inference for better GPU utilization.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
from tqdm import tqdm


class ModelInference:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda:0",
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        inference_batch_size: int = 16,
        lora_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.inference_batch_size = inference_batch_size
        self.device = device
        self.lora_path = lora_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )

        # Load LoRA adapter if provided
        if lora_path and lora_path.lower() != "none":
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print(f"Loaded LoRA adapter from {lora_path}")

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def _generate_single_batch(self, prompts: List[str]) -> List[str]:
        """Generate one response per prompt in a single batch (internal use)."""
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

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate one response per prompt, with automatic batching and progress bar."""
        if len(prompts) <= self.inference_batch_size:
            # Small batch, process directly
            return self._generate_single_batch(prompts)

        # Large batch: split and process with progress bar
        all_responses = []
        total_batches = (len(prompts) + self.inference_batch_size - 1) // self.inference_batch_size

        for i in tqdm(range(0, len(prompts), self.inference_batch_size),
                      total=total_batches, desc="Batch inference"):
            batch = prompts[i:i + self.inference_batch_size]
            responses = self._generate_single_batch(batch)
            all_responses.extend(responses)

        return all_responses

    def generate(self, prompt: str, num_samples: int = 1) -> List[str]:
        """Generate multiple responses for a single prompt."""
        prompts = [prompt] * num_samples
        return self.generate_batch(prompts)

    def batch_inference(
        self,
        samples: List[Dict],
        num_trials: int = 5,
        prompt_formatter: callable = None,
    ) -> List[Dict]:
        """
        Run batch inference on samples.

        Args:
            samples: List of samples with 'question' field
            num_trials: Number of responses per question
            prompt_formatter: Function to format question into prompt

        Returns:
            Samples with added 'responses' field
        """
        if prompt_formatter is None:
            prompt_formatter = lambda s: f"Question: {s['question']}\nAnswer:"

        # Create all prompts: samples × trials
        all_prompts = []
        for sample in samples:
            prompt = prompt_formatter(sample)
            all_prompts.extend([prompt] * num_trials)

        print(f"Processing {len(samples)} samples × {num_trials} trials = {len(all_prompts)} prompts")

        # Batch generate all responses (generate_batch handles batching internally)
        all_responses = self.generate_batch(all_prompts)

        # Group responses back to samples
        results = []
        for i, sample in enumerate(samples):
            start = i * num_trials
            end = start + num_trials
            result = sample.copy()
            result["responses"] = all_responses[start:end]
            results.append(result)

        return results
