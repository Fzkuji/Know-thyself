# Project Instructions for Claude

## Workflow Preferences

1. **Auto-commit after code changes**: After completing code modifications, automatically commit and push to GitHub without asking for confirmation.
   - Use descriptive commit messages
   - Push to the current branch

## Project Structure

- `Know-thyself/`: Main project directory for multi-phase metacognition training
  - `scripts/`: Execution scripts (run_multiphase_ddp.py, etc.)
  - `src/`: Core modules (trainer, inference, evaluator, etc.)

## Key Design Decisions

- Always use full fine-tuning (no LoRA)
- DDP training with 8 GPUs
- Adaptive training: train each sample until learned
- Auto-resume: Pipeline automatically continues from last completed phase
