# Project Instructions for Claude

## Workflow Preferences

1. **IMPORTANT - Auto-commit after EVERY code change**:
   - After completing ANY code modification, ALWAYS commit and push to GitHub immediately
   - Do NOT ask for confirmation, just do it
   - Use descriptive commit messages in English
   - Push to the current branch (usually `main`)
   - This applies to ALL code changes, no exceptions

## Project Structure

- `Know-thyself/`: Main project directory for multi-phase metacognition training
  - `scripts/`: Execution scripts (run_multiphase_ddp.py, etc.)
  - `src/`: Core modules (trainer, inference, evaluator, etc.)

## Key Design Decisions

- Always use full fine-tuning (no LoRA)
- DDP training with 8 GPUs
- Adaptive training: train each sample until learned
- Auto-resume: Pipeline automatically continues from last completed phase

## Common Commands

### Judgment Training (HotpotQA)
```bash
python scripts/run_judgment_training.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --epochs 10 \
    --num_gpus 8 \
    --num_samples 20000 \
    --num_val_samples 5000 \
    --label_mode binary \
    --lr 4e-6 \
    --dataset hotpotqa
```

### Judgment Training (TriviaQA)
```bash
python scripts/run_judgment_training.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --epochs 10 \
    --num_gpus 8 \
    --num_samples 20000 \
    --num_val_samples 5000 \
    --label_mode binary \
    --lr 4e-6 \
    --dataset triviaqa
```

## Hyperparameter Notes

- **Learning Rate**: Use `4e-6` as default
  - `1e-5`: Too large, may degrade model performance
  - `1e-6`: Too small, training is too slow
  - `4e-6`: Good balance between speed and stability
- **Sample Size**: 20000 train / 5000 val is the standard configuration
- **Epochs**: 10 epochs is usually sufficient for convergence
