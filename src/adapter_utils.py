"""
LoRA Adapter Utilities - Merge, load, and manage adapters.

Key strategies:
1. MERGE_INTO_BASE: Merge adapter weights into base model (recommended for knowledge)
2. LOAD_ADAPTER: Load adapter without merging (for judgment)
"""

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_adapter_into_base(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    device_map: str = "auto",
) -> str:
    """
    Merge LoRA adapter into base model weights and save.

    This creates a full model with adapter weights merged - no separate adapter needed.
    Recommended for knowledge learning: merge knowledge into base, then train judgment on top.

    Args:
        base_model_name: HuggingFace model name or path
        adapter_path: Path to LoRA adapter
        output_path: Where to save merged model

    Returns:
        Path to merged model
    """
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter weights into base model...")
    merged_model = model.merge_and_unload()

    # Save merged model
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(str(output_path))

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(str(output_path))

    print("Merge completed!")
    return str(output_path)


def load_model_with_adapter(
    base_model_name: str,
    adapter_path: str = None,
    device_map: str = "auto",
) -> tuple:
    """
    Load model optionally with LoRA adapter.

    Args:
        base_model_name: HuggingFace model name or local path (can be merged model)
        adapter_path: Optional path to LoRA adapter
        device_map: Device mapping

    Returns:
        (model, tokenizer)
    """
    print(f"Loading model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def get_trainable_params(model) -> dict:
    """
    Get count of trainable vs total parameters.

    Returns:
        Dict with trainable_params, total_params, trainable_percent
    """
    trainable = 0
    total = 0

    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": 100 * trainable / total if total > 0 else 0,
    }
