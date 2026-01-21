"""Debug script to check tokenization for training."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from src.label_generator import build_training_sample

def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test sample
    question = "What is the capital of France?"
    ability = "can"

    sample = build_training_sample(question, ability, label_mode="binary")

    print("=" * 60)
    print("Messages:")
    for msg in sample["messages"]:
        print(f"  [{msg['role']}]: {msg['content'][:100]}...")

    # Full text
    full_text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    print("\n" + "=" * 60)
    print("Full text:")
    print(repr(full_text))

    # Prompt only (without assistant response)
    messages_without_assistant = sample["messages"][:-1]
    prompt_text = tokenizer.apply_chat_template(
        messages_without_assistant,
        tokenize=False,
        add_generation_prompt=True
    )
    print("\n" + "=" * 60)
    print("Prompt text (without assistant response):")
    print(repr(prompt_text))

    # Tokenize both
    full_tokens = tokenizer(full_text, truncation=True, max_length=512, padding=False)
    prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=512, padding=False)

    input_ids = full_tokens["input_ids"]
    prompt_len = len(prompt_tokens["input_ids"])

    print("\n" + "=" * 60)
    print(f"Full tokens length: {len(input_ids)}")
    print(f"Prompt tokens length: {prompt_len}")
    print(f"Response tokens length: {len(input_ids) - prompt_len}")

    # Check if response tokens exist
    if len(input_ids) <= prompt_len:
        print("\n⚠️  WARNING: No response tokens! prompt_len >= full_len")
        print("This would cause empty labels and potentially high loss!")

    # Labels
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    print("\n" + "=" * 60)
    print("Labels analysis:")
    print(f"  Total labels: {len(labels)}")
    print(f"  Masked (-100): {labels.count(-100)}")
    print(f"  Non-masked: {len(labels) - labels.count(-100)}")

    # Decode response part
    if len(input_ids) > prompt_len:
        response_tokens = input_ids[prompt_len:]
        response_text = tokenizer.decode(response_tokens)
        print(f"\n  Response tokens: {response_tokens}")
        print(f"  Response text: {repr(response_text)}")
    else:
        print("\n  ⚠️  No response tokens to decode!")

    # Check alignment
    print("\n" + "=" * 60)
    print("Alignment check:")
    print(f"  Full text ends with: {repr(full_text[-50:])}")
    print(f"  Prompt text ends with: {repr(prompt_text[-50:])}")

    # Check if prompt_text is a prefix of full_text
    if full_text.startswith(prompt_text):
        print("  ✓ Prompt is prefix of full text")
    else:
        print("  ⚠️  Prompt is NOT a prefix of full text!")
        # Find where they differ
        for i, (a, b) in enumerate(zip(full_text, prompt_text)):
            if a != b:
                print(f"  First difference at position {i}:")
                print(f"    Full: {repr(full_text[max(0,i-20):i+20])}")
                print(f"    Prompt: {repr(prompt_text[max(0,i-20):i+20])}")
                break


def test_data_collator():
    """Test data collator behavior."""
    import torch
    from dataclasses import dataclass
    from typing import Any, Dict, List

    @dataclass
    class DataCollatorForCausalLM:
        """Data collator for causal language modeling."""
        tokenizer: Any
        padding: bool = True
        max_length: int = None
        pad_to_multiple_of: int = None
        label_pad_token_id: int = -100

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            labels = [feature.pop("labels") for feature in features] if "labels" in features[0] else None

            batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            if labels is not None:
                max_label_length = max(len(l) for l in labels)
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                # Match padding side with tokenizer
                pad_left = getattr(self.tokenizer, "padding_side", "right") == "left"

                padded_labels = []
                for label in labels:
                    remainder = max_label_length - len(label)
                    if pad_left:
                        # Left padding: add -100 at the beginning
                        padded_label = [self.label_pad_token_id] * remainder + list(label)
                    else:
                        # Right padding: add -100 at the end
                        padded_label = list(label) + [self.label_pad_token_id] * remainder
                    padded_labels.append(padded_label)

                batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

            return batch

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Same as training script

    print("\n" + "=" * 60)
    print("Testing DataCollator with padding_side='left'")
    print("=" * 60)

    # Create two samples with different lengths
    samples = [
        {"question": "What is 2+2?", "ability": "can"},
        {"question": "What is the capital of France and what is its population?", "ability": "cannot"},
    ]

    features = []
    for s in samples:
        sample = build_training_sample(s["question"], s["ability"], label_mode="binary")
        full_text = tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)
        messages_without_assistant = sample["messages"][:-1]
        prompt_text = tokenizer.apply_chat_template(messages_without_assistant, tokenize=False, add_generation_prompt=True)

        full_tokens = tokenizer(full_text, truncation=True, max_length=512, padding=False)
        prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=512, padding=False)

        input_ids = full_tokens["input_ids"]
        prompt_len = len(prompt_tokens["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        features.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        })
        print(f"\nSample: {s['question'][:30]}...")
        print(f"  input_ids length: {len(input_ids)}")
        print(f"  labels length: {len(labels)}")
        print(f"  Non-masked labels: {len([l for l in labels if l != -100])}")

    collator = DataCollatorForCausalLM(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    batch = collator(features)

    print("\n" + "-" * 40)
    print("After collation:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")

    # Check alignment
    for i in range(len(samples)):
        input_ids = batch["input_ids"][i]
        labels = batch["labels"][i]
        attn_mask = batch["attention_mask"][i]

        # Find where actual content starts (not padding)
        content_start = (attn_mask == 1).nonzero(as_tuple=True)[0][0].item() if (attn_mask == 1).any() else 0

        print(f"\n  Sample {i}:")
        print(f"    Content starts at: {content_start}")
        print(f"    Padding tokens: {content_start}")

        # Check if labels align with input_ids
        non_masked_labels = [(j, l.item()) for j, l in enumerate(labels) if l != -100]
        print(f"    Non-masked label positions: {[p for p, _ in non_masked_labels]}")

        if non_masked_labels:
            for pos, label_id in non_masked_labels[:3]:  # Show first 3
                input_id = input_ids[pos].item()
                if input_id == label_id:
                    print(f"      ✓ Position {pos}: input={input_id}, label={label_id} (match)")
                else:
                    print(f"      ⚠️  Position {pos}: input={input_id}, label={label_id} (MISMATCH!)")


def test_loss_computation():
    """Test that loss would be computed correctly."""
    import torch
    import torch.nn.functional as F

    print("\n" + "=" * 60)
    print("Testing Loss Computation Simulation")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Create sample
    sample = build_training_sample("What is 2+2?", "can", label_mode="binary")
    full_text = tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)
    messages_without_assistant = sample["messages"][:-1]
    prompt_text = tokenizer.apply_chat_template(messages_without_assistant, tokenize=False, add_generation_prompt=True)

    full_tokens = tokenizer(full_text, truncation=True, max_length=512, padding=False)
    prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=512, padding=False)

    input_ids = full_tokens["input_ids"]
    prompt_len = len(prompt_tokens["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    print(f"Input IDs: {input_ids}")
    print(f"Labels: {labels}")
    print(f"Prompt length: {prompt_len}")
    print(f"Response tokens: {input_ids[prompt_len:]}")

    # Simulate model output (random logits)
    vocab_size = 151936  # Qwen vocab size
    seq_len = len(input_ids)

    # Create fake logits
    torch.manual_seed(42)
    logits = torch.randn(1, seq_len, vocab_size)

    # Create labels tensor
    labels_tensor = torch.tensor([labels])

    # Shift for causal LM loss (predict next token)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels_tensor[..., 1:].contiguous()

    print(f"\nShift logits shape: {shift_logits.shape}")
    print(f"Shift labels shape: {shift_labels.shape}")

    # Count how many tokens will contribute to loss
    non_masked = (shift_labels != -100).sum().item()
    print(f"Non-masked tokens for loss: {non_masked}")

    # Compute loss
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100
    )

    print(f"\nSimulated loss: {loss.item():.4f}")

    # Expected loss for random predictions
    expected_random_loss = torch.log(torch.tensor(vocab_size)).item()
    print(f"Expected random loss (log(vocab_size)): {expected_random_loss:.4f}")

    if loss.item() < expected_random_loss * 2:
        print("✓ Loss is in reasonable range for random logits")
    else:
        print("⚠️  Loss seems too high!")

    # Check that the right positions are being used
    print("\nPositions contributing to loss:")
    for i, label in enumerate(shift_labels[0].tolist()):
        if label != -100:
            token_text = tokenizer.decode([label])
            print(f"  Position {i}: label={label} ({repr(token_text)})")


if __name__ == "__main__":
    main()
    test_data_collator()
    test_loss_computation()
