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

## Project Structure

```
Know-thyself/
├── src/
│   ├── data_loader.py      # Load TriviaQA dataset
│   ├── inference.py        # Batch inference
│   ├── evaluator.py        # Evaluate responses
│   ├── label_generator.py  # Generate training labels
│   ├── dataset_builder.py  # Build HF datasets
│   ├── trainer.py          # LoRA fine-tuning
│   ├── knowledge_trainer.py # QA data building
│   ├── adapter_utils.py    # LoRA merge utilities
│   └── pipeline.py         # Multi-phase state management
├── scripts/
│   ├── step1_collect_responses.py
│   ├── step2_build_dataset.py
│   ├── step3_train.py
│   ├── step4_evaluate.py
│   ├── phase2_knowledge.py
│   ├── phase3_update_judgment.py
│   └── run_multiphase.py   # Unified entry point
├── experiments/            # Experiment outputs
└── run_multiphase_pipeline.sh
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Recommended)

```bash
# Run full 3-phase pipeline
bash run_multiphase_pipeline.sh Qwen/Qwen2.5-0.5B-Instruct 10000 1000 triviaqa 512 32
```

Parameters:
1. `model` - Model name (default: Qwen/Qwen2.5-0.5B-Instruct)
2. `train_samples` - Number of training samples (default: 1000)
3. `test_samples` - Number of test samples (default: 100)
4. `dataset` - Dataset name for experiment naming (default: triviaqa)
5. `inference_batch_size` - Batch size for inference (default: 16)
6. `train_batch_size` - Batch size for training (default: 32)

### Python API

```bash
# Run all phases
python scripts/run_multiphase.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --num_samples 10000 \
    --test_samples 1000 \
    --inference_batch_size 512 \
    --batch_size 32

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
│   └── lora_judgment_v1/
├── phase2_knowledge/
│   ├── qa_training_data.jsonl
│   ├── lora_knowledge/
│   └── base_with_knowledge/
└── phase3_judgment/
    ├── responses_post_knowledge.jsonl
    ├── training_data_v2.jsonl
    └── lora_judgment_v2/
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
