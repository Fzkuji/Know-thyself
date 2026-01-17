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
