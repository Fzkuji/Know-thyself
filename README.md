# Know-thyself

Fine-tuning LLMs to develop metacognitive abilities - knowing what they know and don't know before making predictions.

## Overview

This project implements a **three-phase training pipeline** to enhance LLMs' self-awareness:

1. **Phase 1**: Train judgment ability (learn what you don't know)
2. **Phase 2**: Learn knowledge (actually learn the answers)
3. **Phase 3**: Update judgment (now you know → become confident)

The goal is to train models that can accurately predict whether they will be able to correctly answer a given question **before** attempting to answer it.

## Method

### Three-Phase Pipeline

```
Phase 1: Initial Judgment Training
├── 1.1 Collect responses (train split, 5 trials per question)
├── 1.2 Build judgment training data
├── 1.3 Evaluate baseline (before training)
├── 1.4 Train judgment → LoRA_judgment_v1
└── 1.5 Evaluate (train + validation splits)

Phase 2: Knowledge Learning
├── 2.1 Build QA training data
├── 2.2 Train knowledge → LoRA_knowledge
├── 2.3 Merge into base → base_with_knowledge
└── 2.4 Test knowledge acquisition (train + validation)

Phase 3: Update Judgment
├── 3.1 Re-collect responses with knowledge model
├── 3.2 Build updated judgment data
├── 3.3 Train judgment v2 → LoRA_judgment_v2
└── 3.4 Final evaluation (train + validation)
```

### Ability Classification

Based on accuracy across 5 trials:
- **100% correct** → "can"
- **50%-99% correct** → "uncertain"
- **<50% correct** → "cannot"

### Data Splits

- **Training**: TriviaQA `train` split
- **Testing**: TriviaQA `validation` split (held-out)

## Supported Datasets

| Dataset | Split | Size | Description |
|---------|-------|------|-------------|
| TriviaQA | train | 138K | Trivia questions with multiple answer aliases |
| TriviaQA | validation | 18K | Held-out test set |
| HotpotQA | train | 90K | Multi-hop reasoning questions |
| HotpotQA | validation | 7.4K | Held-out test set |

## Project Structure

```
Know-thyself/
├── src/
│   ├── data_loader.py      # Load QA datasets (TriviaQA, HotpotQA)
│   ├── inference.py        # Batch inference
│   ├── evaluator.py        # Evaluate responses
│   ├── label_generator.py  # Generate training labels
│   ├── dataset_builder.py  # Build HF datasets
│   ├── trainer.py          # LoRA fine-tuning
│   ├── knowledge_trainer.py # QA data building
│   ├── adapter_utils.py    # LoRA merge utilities
│   └── pipeline.py         # Multi-phase state management
├── scripts/
│   ├── run_judgment_training.py  # Main entry point (recommended)
│   ├── inference_ddp.py          # Multi-GPU inference
│   ├── train_deepspeed.py        # DeepSpeed ZeRO-3 training
│   ├── run_multiphase.py         # Legacy multi-phase pipeline
│   └── debug_tokenization.py     # Debug utilities
├── configs/
│   └── ds_config_zero3.json      # DeepSpeed config
├── experiments/            # Experiment outputs
└── run_multiphase_pipeline.sh
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Judgment Training Pipeline (Recommended)

This is the **simplified single-phase pipeline** for training judgment ability with multi-GPU support.

```bash
# Qwen2.5-7B + HotpotQA (with context)
python scripts/run_judgment_training.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset hotpotqa \
    --epochs 10 \
    --num_gpus 8 \
    --num_samples 20000 \
    --num_val_samples 5000 \
    --label_mode binary \
    --lr 4e-6 \
    --batch_size 16

# Qwen2.5-7B + TriviaQA (closed-book)
python scripts/run_judgment_training.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset triviaqa \
    --epochs 10 \
    --num_gpus 8 \
    --num_samples 20000 \
    --num_val_samples 5000 \
    --label_mode binary \
    --lr 4e-6 \
    --batch_size 16

# GPT-OSS-20B + HotpotQA (with context, smaller batch size for 20B model)
python scripts/run_judgment_training.py \
    --model /data/public/llms/GPT-OSS/gpt-oss-20b \
    --dataset hotpotqa \
    --epochs 10 \
    --num_gpus 8 \
    --num_samples 20000 \
    --num_val_samples 5000 \
    --label_mode binary \
    --lr 4e-6 \
    --batch_size 2

# GPT-OSS-20B + TriviaQA (closed-book)
python scripts/run_judgment_training.py \
    --model /data/public/llms/GPT-OSS/gpt-oss-20b \
    --dataset triviaqa \
    --epochs 10 \
    --num_gpus 8 \
    --num_samples 20000 \
    --num_val_samples 5000 \
    --label_mode binary \
    --lr 4e-6 \
    --batch_size 2
```

**Hyperparameter Notes:**
- **Learning Rate**: `4e-6` recommended (1e-5 too large degrades performance, 1e-6 too slow)
- **Batch Size**: 16 for 7B models, 2 for 20B models (to avoid OOM with long contexts)
- **Sample Size**: 20000 train / 5000 val is standard configuration
- **Datasets**: HotpotQA uses context (open-book reading comprehension), TriviaQA is closed-book (tests world knowledge)

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Qwen/Qwen2.5-7B-Instruct | Model name or path |
| `--dataset` | triviaqa | Dataset: `triviaqa` or `hotpotqa` |
| `--epochs` | 10 | Number of training epochs |
| `--num_samples` | 5000 | Number of training samples |
| `--num_val_samples` | 1000 | Number of validation samples |
| `--lr` | 1e-5 | Learning rate |
| `--batch_size` | 16 | Batch size per GPU |
| `--num_gpus` | 8 | Number of GPUs |
| `--label_mode` | binary | Label mode: `binary` (yes/no) or `uncertainty` (yes/uncertain/no) |
| `--output_dir` | auto | Output directory (auto-generated if not specified) |
| `--skip_collect` | false | Skip response collection (use existing) |
| `--start_epoch` | 1 | Resume from this epoch |

**Pipeline Steps:**

```
Step 0: Evaluate pretrained model
├── 0.1 Collect train responses (get ability labels)
├── 0.2 Test pretrained judgment (Train)
├── 0.3 Collect validation responses
└── 0.4 Test pretrained judgment (Val)

For each epoch N:
├── N.1 Train on samples where judgment was wrong
├── N.2 Re-collect responses (get new ability labels)
├── N.3 Test judgment accuracy (Train)
├── N.4 Re-collect validation responses
└── N.5 Test judgment accuracy (Val)
```

**Output Structure:**

```
experiments/qwen2.5_7b_instruct_triviaqa_binary_lr1e05_20240121_123456/
├── responses.jsonl              # Train set ability labels
├── val_responses.jsonl          # Val set ability labels
├── tested_epoch0.jsonl          # Pretrained judgment results (Train)
├── val_tested_epoch0.jsonl      # Pretrained judgment results (Val)
├── epoch_1/                     # Trained model checkpoint
├── responses_epoch1.jsonl       # Updated ability labels
├── tested_epoch1.jsonl          # Epoch 1 judgment results
├── ...
└── epoch_N/                     # Final model
```

**Summary Output:**

```
Epoch    Train Judg   Train QA     Train AUROC  Val Judg     Val QA       Val AUROC
----------------------------------------------------------------------------------
0        52.3%        45.2%        0.5234       51.8%        44.9%        0.5198
1        68.5%        45.1%        0.7123       65.2%        44.8%        0.6892
...
```

---

### Multi-phase Pipeline (Legacy)

### Quick Start (Recommended)

```bash
# Show help
bash run_multiphase_pipeline.sh --help

# Run full 3-phase pipeline (LoRA fine-tuning, lr=1e-4)
bash run_multiphase_pipeline.sh \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --train_samples 10000 \
    --test_samples 1000 \
    --inference_batch 512 \
    --train_batch 32

# Run full 3-phase pipeline (Full fine-tuning, lr=1e-5)
bash run_multiphase_pipeline.sh \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --train_samples 10000 \
    --test_samples 1000 \
    --no_lora
```

Parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Qwen/Qwen2.5-0.5B-Instruct | Model name |
| `--train_samples` | 1000 | Number of training samples |
| `--test_samples` | 100 | Number of test samples |
| `--dataset` | triviaqa | Dataset name for experiment naming |
| `--inference_batch` | 16 | Batch size for inference |
| `--train_batch` | 32 | Batch size for training |
| `--epochs` | 1 | Epochs for judgment training |
| `--knowledge_epochs` | 5 | Epochs for knowledge training |
| `--num_trials` | 5 | Responses per question |
| `--no_lora` | false | Use full fine-tuning instead of LoRA |

### Python API

```bash
# Run all phases (LoRA, lr=1e-4)
python scripts/run_multiphase.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --num_samples 10000 \
    --test_samples 1000 \
    --inference_batch_size 512 \
    --batch_size 32

# Run all phases (Full fine-tuning, lr=1e-5)
python scripts/run_multiphase.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --num_samples 10000 \
    --test_samples 1000 \
    --inference_batch_size 512 \
    --batch_size 32 \
    --no_lora \
    --lr 1e-5

# Run specific phase
python scripts/run_multiphase.py --phase 2

# Resume from checkpoint
python scripts/run_multiphase.py --experiment <name> --resume
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Learning rate for LoRA training (1e-5 for full fine-tuning) |
| `epochs` | 1 | Epochs for judgment training |
| `knowledge_epochs` | 5 | Epochs for knowledge training |
| `batch_size` | 4/32 | Training batch size |
| `inference_batch_size` | 16 | Inference batch size |
| `num_trials` | 5 | Responses per question |
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA alpha |

## Experiment Outputs

Experiments are saved to `experiments/{model}_{dataset}_train{N}_test{M}_{timestamp}/`:

```
experiments/Qwen2.5-0.5B_triviaqa_train10000_test1000_0114_1600/
├── config.json
├── pipeline_state.json
├── phase1_judgment/
│   ├── responses.jsonl
│   ├── training_data.jsonl
│   └── judgment_v1/          # LoRA adapter or full model
├── phase2_knowledge/
│   ├── qa_training_data.jsonl
│   ├── knowledge/            # LoRA adapter or full model
│   └── base_with_knowledge/  # Merged model (LoRA only)
└── phase3_judgment/
    ├── responses_post_knowledge.jsonl
    ├── training_data_v2.jsonl
    └── judgment_v2/          # LoRA adapter or full model
```

## Evaluation Metrics

Each phase evaluates on both train and validation splits:

- **Exact Match**: Predicted ability matches actual ability
- **Confusion Matrix**: 3x3 (can/uncertain/cannot)
- **Ability Distribution**: Predicted vs Actual

## Expected Results

- **Phase 1**: Model learns to judge (may become conservative)
- **Phase 2**: Model learns knowledge (can answer more questions)
- **Phase 3**: Model updates judgment (becomes confident and accurate)

## License

MIT
