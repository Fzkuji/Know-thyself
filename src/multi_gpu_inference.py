"""
Multi-GPU Inference Module - Parallel inference across multiple GPUs.

Each GPU loads a separate model instance, and data is distributed across GPUs.
This is more efficient than DDP for inference (no gradient sync needed).

Usage:
    inference = MultiGPUInference(model_name="Qwen/Qwen2.5-7B-Instruct")
    results = inference.batch_inference(samples, num_trials=5)
"""

import os
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
from tqdm import tqdm
from queue import Empty
import time


def _worker_init(
    rank: int,
    world_size: int,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    ready_event: mp.Event,
):
    """
    Worker process: load model on specific GPU and process tasks from queue.
    """
    # Set device for this worker
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    print(f"[Worker {rank}] Loading model on {device}...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[Worker {rank}] Model loaded, ready to process")
    ready_event.set()  # Signal that this worker is ready

    # Process tasks from queue
    while True:
        try:
            task = input_queue.get(timeout=1.0)

            if task is None:  # Poison pill - shutdown signal
                print(f"[Worker {rank}] Shutting down")
                break

            task_id, prompts = task

            # Generate responses
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            responses = []
            for i, output in enumerate(outputs):
                input_len = (inputs["attention_mask"][i] == 1).sum().item()
                response = tokenizer.decode(
                    output[input_len:],
                    skip_special_tokens=True
                ).strip()
                responses.append(response)

            output_queue.put((task_id, responses))

        except Empty:
            continue
        except Exception as e:
            print(f"[Worker {rank}] Error: {e}")
            continue

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()


class MultiGPUInference:
    """
    Multi-GPU inference with one model per GPU.

    Data is distributed round-robin across GPUs for parallel processing.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        inference_batch_size: int = 16,
        num_gpus: int = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.inference_batch_size = inference_batch_size

        # Detect available GPUs
        if num_gpus is None:
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = min(num_gpus, torch.cuda.device_count())

        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available for multi-GPU inference")

        print(f"MultiGPUInference: Using {self.num_gpus} GPUs")

        # Initialize multiprocessing
        mp.set_start_method('spawn', force=True)

        self.workers = []
        self.input_queues = []
        self.output_queue = mp.Queue()
        self.ready_events = []

        # Start worker processes
        for rank in range(self.num_gpus):
            input_queue = mp.Queue()
            ready_event = mp.Event()

            worker = mp.Process(
                target=_worker_init,
                args=(
                    rank,
                    self.num_gpus,
                    model_name,
                    max_new_tokens,
                    temperature,
                    input_queue,
                    self.output_queue,
                    ready_event,
                )
            )
            worker.start()

            self.workers.append(worker)
            self.input_queues.append(input_queue)
            self.ready_events.append(ready_event)

        # Wait for all workers to be ready
        print("Waiting for all workers to load models...")
        for i, event in enumerate(self.ready_events):
            event.wait()
            print(f"  Worker {i} ready")
        print("All workers ready!")

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Shutdown all worker processes."""
        if hasattr(self, 'workers'):
            # Send poison pills
            for queue in self.input_queues:
                try:
                    queue.put(None)
                except:
                    pass

            # Wait for workers to finish
            for worker in self.workers:
                worker.join(timeout=5.0)
                if worker.is_alive():
                    worker.terminate()

            self.workers = []

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate one response per prompt, distributed across GPUs."""
        # Split prompts into batches
        batches = []
        for i in range(0, len(prompts), self.inference_batch_size):
            batch = prompts[i:i + self.inference_batch_size]
            batches.append(batch)

        # Distribute batches to workers round-robin
        results = {}
        pending = 0

        for task_id, batch in enumerate(batches):
            worker_id = task_id % self.num_gpus
            self.input_queues[worker_id].put((task_id, batch))
            pending += 1

        # Collect results
        while pending > 0:
            task_id, responses = self.output_queue.get()
            results[task_id] = responses
            pending -= 1

        # Reconstruct in order
        all_responses = []
        for task_id in range(len(batches)):
            all_responses.extend(results[task_id])

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
        Run batch inference on samples using multiple GPUs.

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
        print(f"Using {self.num_gpus} GPUs with batch size {self.inference_batch_size}")

        # Batch generate all responses
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


# Convenience function for scripts
def create_inference(
    model_name: str,
    inference_batch_size: int = 16,
    temperature: float = 1.0,
    multi_gpu: bool = False,
    num_gpus: int = None,
):
    """
    Create inference instance.

    Args:
        model_name: Model to load
        inference_batch_size: Batch size for inference
        temperature: Sampling temperature
        multi_gpu: Use multi-GPU mode (each GPU loads one model)
        num_gpus: Number of GPUs to use (None=all available)

    Returns:
        ModelInference or MultiGPUInference instance
    """
    available_gpus = torch.cuda.device_count()

    if multi_gpu and available_gpus > 1:
        return MultiGPUInference(
            model_name=model_name,
            inference_batch_size=inference_batch_size,
            temperature=temperature,
            num_gpus=num_gpus,
        )
    else:
        # Single GPU mode: load model on cuda:0
        from src.inference import ModelInference
        return ModelInference(
            model_name=model_name,
            inference_batch_size=inference_batch_size,
            temperature=temperature,
            device="cuda:0",
        )


if __name__ == "__main__":
    # Test multi-GPU inference
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    print(f"Testing multi-GPU inference with {args.model}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    # Create test samples
    samples = [
        {"question": "What is the capital of France?"},
        {"question": "What is 2 + 2?"},
        {"question": "Who wrote Romeo and Juliet?"},
        {"question": "What is the largest planet in our solar system?"},
    ]

    # Create inference
    inference = create_inference(
        model_name=args.model,
        inference_batch_size=args.batch_size,
        num_gpus=args.num_gpus,
    )

    # Run inference
    print("\nRunning inference...")
    results = inference.batch_inference(samples, num_trials=3)

    # Print results
    for r in results:
        print(f"\nQ: {r['question']}")
        for i, resp in enumerate(r['responses']):
            print(f"  A{i+1}: {resp[:100]}...")

    # Cleanup
    if hasattr(inference, 'shutdown'):
        inference.shutdown()

    print("\nDone!")
