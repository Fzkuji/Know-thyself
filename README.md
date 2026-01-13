# Know-thyself

Fine-tuning LLMs to develop metacognitive abilities - knowing what they know and don't know before making predictions.

## Overview

This project aims to enhance LLMs' self-awareness of their own capabilities. The goal is to train models that can accurately predict whether they will be able to correctly answer a given question **before** attempting to answer it.

## Method

**Four-step pipeline:**

1. **Collect Responses** (train split): Query the model 5 times per question
2. **Generate Labels**: Classify ability based on accuracy
   - 100% correct → "can"
   - 50%-99% correct → "uncertain"
   - <50% correct → "cannot"
3. **Train Metacognition**: Fine-tune model to predict its own ability
4. **Evaluate** (test split): Compare predicted vs actual ability

## Project Structure

```
Know-thyself/
├── src/
│   ├── data_loader.py      # Load TriviaQA dataset
│   ├── inference.py        # Model inference (N trials)
│   ├── evaluator.py        # Evaluate responses
│   ├── label_generator.py  # Generate training labels
│   ├── dataset_builder.py  # Build HF datasets
│   └── trainer.py          # LoRA fine-tuning
├── scripts/
│   ├── step1_collect_responses.py
│   ├── step2_build_dataset.py
│   ├── step3_train.py
│   └── step4_evaluate.py
├── configs/
│   └── config.yaml
├── data/                   # Generated data
├── outputs/                # Trained models
└── run.py                  # Main entry point
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run full training pipeline (steps 1-3)
python run.py --step all --model Qwen/Qwen2.5-7B-Instruct --num_samples 1000

# Or run steps individually
python run.py --step 1 --num_samples 1000    # Collect responses (train split)
python run.py --step 2                        # Build dataset
python run.py --step 3 --epochs 3             # Train
python run.py --step 4 --test_samples 100     # Evaluate (test split)

# Use different splits
python run.py --step 1 --train_split train    # Use train split for training
python run.py --step 4 --test_split test      # Use test split for evaluation
```

## Evaluation Metrics

Step 4 outputs:
- **Exact Match**: Predicted ability matches actual ability
- **Accuracy**: (TP + TN) / Total
- **Precision/Recall/F1**: For "can" predictions
- **Overconfident**: Predicted "can" but actually "cannot"
- **Underconfident**: Predicted "cannot" but actually "can"

## License

MIT
