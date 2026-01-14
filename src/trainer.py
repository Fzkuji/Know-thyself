"""
Trainer module - fine-tune model for metacognition.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from typing import Optional
from datasets import Dataset


def setup_model_for_training(
    model_name: str,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> tuple:
    """
    Setup model and tokenizer for training.

    Args:
        model_name: HuggingFace model name or local path (e.g., merged model)
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for stability
        device_map="auto",
        trust_remote_code=True,
    )

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def train_metacognition(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    output_dir: str = "./outputs/metacog",
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-4,  # 1e-4 for LoRA, 1e-5 for full fine-tuning
    max_length: int = 512,
    use_lora: bool = True,
):
    """
    Train model for metacognition task using SFTTrainer.

    SFTTrainer properly handles instruction tuning by only computing loss
    on the assistant's response, not the prompt.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset with 'text' field (formatted conversations)
        val_dataset: Validation dataset
        output_dir: Output directory
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_length: Max sequence length
        use_lora: Whether using LoRA
    """
    # Use bf16 for mixed precision (works for both LoRA and full fine-tuning)
    bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    # SFTConfig extends TrainingArguments with SFT-specific options
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",  # Don't save checkpoints during training
        eval_strategy="epoch" if val_dataset else "no",
        bf16=bf16,
        report_to="none",
        max_grad_norm=1.0,  # Gradient clipping
        weight_decay=0.01,  # Regularization
        max_length=max_length,  # Max sequence length
        dataset_text_field="text",  # Field containing the formatted text
        packing=False,  # Don't pack multiple samples
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)

    return trainer
