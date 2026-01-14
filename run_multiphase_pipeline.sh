#!/bin/bash
# Know-thyself: Multi-phase Pipeline
# Complete "learn -> know -> confident" cycle
#
# Phase 1: Initial judgment training (learn what you don't know)
# Phase 2: Knowledge learning (actually learn the knowledge)
# Phase 3: Update judgment (now you know -> become confident)
#
# Usage: bash run_multiphase_pipeline.sh [model] [train_samples] [test_samples] [dataset] [inference_batch_size] [train_batch_size] [no_lora]
# Experiment name is auto-generated: {model}_{dataset}_train{N}_test{M}_{timestamp}

set -e  # Exit on error

# Default parameters
MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
TRAIN_SAMPLES=${2:-1000}
TEST_SAMPLES=${3:-100}
DATASET=${4:-"triviaqa"}
INFERENCE_BATCH_SIZE=${5:-16}
TRAIN_BATCH_SIZE=${6:-32}
NO_LORA=${7:-"false"}  # Set to "true" for full fine-tuning

EPOCHS=1
KNOWLEDGE_EPOCHS=5
NUM_TRIALS=5

# Set learning rate based on training mode
if [ "$NO_LORA" = "true" ]; then
    LR="1e-5"
    TRAINING_MODE="Full Fine-tuning"
else
    LR="1e-4"
    TRAINING_MODE="LoRA"
fi

PROJECT_ROOT=$(dirname "$0")

echo "=============================================="
echo "Know-thyself Multi-phase Pipeline"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Train samples: $TRAIN_SAMPLES"
echo "Test samples: $TEST_SAMPLES"
echo "Inference batch size: $INFERENCE_BATCH_SIZE"
echo "Training batch size: $TRAIN_BATCH_SIZE"
echo "Training mode: $TRAINING_MODE"
echo "Learning rate: $LR"
echo "(Experiment name will be auto-generated)"
echo "=============================================="

# Build command
CMD="python $PROJECT_ROOT/scripts/run_multiphase.py \
    --model $MODEL \
    --dataset $DATASET \
    --num_samples $TRAIN_SAMPLES \
    --test_samples $TEST_SAMPLES \
    --num_trials $NUM_TRIALS \
    --inference_batch_size $INFERENCE_BATCH_SIZE \
    --epochs $EPOCHS \
    --knowledge_epochs $KNOWLEDGE_EPOCHS \
    --batch_size $TRAIN_BATCH_SIZE \
    --lr $LR"

# Add --no_lora flag if full fine-tuning
if [ "$NO_LORA" = "true" ]; then
    CMD="$CMD --no_lora"
fi

# Run the command
eval $CMD

echo ""
echo "=============================================="
echo "Multi-phase Pipeline completed!"
echo "=============================================="
echo "Results saved to: experiments/{auto-generated-name}/"
echo ""
echo "Expected improvements:"
echo "  Phase 1: Model learns to judge (may become conservative)"
echo "  Phase 2: Model learns knowledge (can answer more questions)"
echo "  Phase 3: Model updates judgment (becomes confident and accurate)"
echo "=============================================="
