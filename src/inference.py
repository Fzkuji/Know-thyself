"""
Model inference module - query model multiple times for each question.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
from tqdm import tqdm


class ModelInference:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "auto",
        max_new_tokens: int = 64,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, num_samples: int = 1) -> List[str]:
        """
        Generate multiple responses for a single prompt.

        Args:
            prompt: Input prompt
            num_samples: Number of responses to generate

        Returns:
            List of generated responses
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        responses = []
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            responses.append(response)

        return responses

    def batch_inference(
        self,
        samples: List[Dict],
        num_trials: int = 5,
        prompt_formatter: callable = None,
    ) -> List[Dict]:
        """
        Run inference on multiple samples, each queried multiple times.

        Args:
            samples: List of samples with 'question' field
            num_trials: Number of times to query each question
            prompt_formatter: Function to format question into prompt

        Returns:
            Samples with added 'responses' field
        """
        if prompt_formatter is None:
            prompt_formatter = lambda q: f"Question: {q}\nAnswer:"

        results = []
        for sample in tqdm(samples, desc="Running inference"):
            prompt = prompt_formatter(sample["question"])
            responses = self.generate(prompt, num_samples=num_trials)

            result = sample.copy()
            result["responses"] = responses
            results.append(result)

        return results
