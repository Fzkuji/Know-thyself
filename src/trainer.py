"""
Trainer module - fine-tune model for metacognition.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, Optional
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
        torch_dtype=torch.float16,
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


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    """Tokenize dataset for training."""

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)


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
    use_lora: bool = True,  # Whether using LoRA (affects mixed precision settings)
):
    """
    Train model for metacognition task.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Output directory
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_length: Max sequence length
        use_lora: Whether using LoRA (fp16 for LoRA, bf16 for full fine-tuning)
    """
    # Tokenize
    train_tokenized = tokenize_dataset(train_dataset, tokenizer, max_length)
    val_tokenized = tokenize_dataset(val_dataset, tokenizer, max_length) if val_dataset else None

    # Mixed precision settings:
    # - LoRA: use fp16 (works well with PEFT)
    # - Full fine-tuning: use bf16 (fp16 causes gradient unscaling errors)
    if use_lora:
        fp16 = True
        bf16 = False
    else:
        # Full fine-tuning: prefer bf16 if available, otherwise disable mixed precision
        fp16 = False
        bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_tokenized else "no",
        fp16=fp16,
        bf16=bf16,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)

    return trainer
