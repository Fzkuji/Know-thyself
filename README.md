# Know-thyself

Fine-tuning LLMs to develop metacognitive abilities - knowing what they know and don't know before making predictions.

## Overview

This project aims to enhance LLMs' self-awareness of their own capabilities. The goal is to train models that can accurately predict whether they will be able to correctly answer a given question **before** attempting to answer it.

## Method

**Three-step training pipeline:**

1. **Collect Responses**: Query the model 5 times per question
2. **Generate Labels**: Classify ability based on accuracy
   - 100% correct → "can"
   - 50%-99% correct → "uncertain"
   - <50% correct → "cannot"
3. **Train Metacognition**: Fine-tune model to predict its own ability

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
│   └── step3_train.py
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
# Run full pipeline
python run.py --step all --model meta-llama/Llama-2-7b-chat-hf

# Or run steps individually
python run.py --step 1 --num_samples 1000  # Collect responses
python run.py --step 2                      # Build dataset
python run.py --step 3 --epochs 3           # Train
```

## License

MIT
