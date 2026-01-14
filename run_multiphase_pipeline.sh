#!/bin/bash
# Know-thyself: Multi-phase Pipeline
# Complete "learn -> know -> confident" cycle
#
# Phase 1: Initial judgment training (learn what you don't know)
# Phase 2: Knowledge learning (actually learn the knowledge)
# Phase 3: Update judgment (now you know -> become confident)
#
# Usage: bash run_multiphase_pipeline.sh --model Qwen/Qwen2.5-0.5B-Instruct --train_samples 10000 --test_samples 1000
# Experiment name is auto-generated: {model}_{dataset}_train{N}_test{M}_{timestamp}

set -e  # Exit on error

# Default parameters
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_SAMPLES=1000
TEST_SAMPLES=100
DATASET="triviaqa"
INFERENCE_BATCH_SIZE=16
TRAIN_BATCH_SIZE=32
NO_LORA="false"
EPOCHS=2
KNOWLEDGE_EPOCHS=2
NUM_TRIALS=5
ADAPTIVE="true"
MAX_STEPS_PER_SAMPLE=10

# Help function
show_help() {
    echo "Usage: bash run_multiphase_pipeline.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model               Model name (default: Qwen/Qwen2.5-0.5B-Instruct)"
    echo "  --train_samples       Number of training samples (default: 1000)"
    echo "  --test_samples        Number of test samples (default: 100)"
    echo "  --dataset             Dataset name for experiment naming (default: triviaqa)"
    echo "  --inference_batch     Batch size for inference (default: 16)"
    echo "  --train_batch         Batch size for training (default: 32)"
    echo "  --epochs              Epochs for judgment training (default: 2)"
    echo "  --knowledge_epochs    Epochs for knowledge training (default: 2)"
    echo "  --num_trials          Responses per question (default: 5)"
    echo "  --max_steps           Max steps per sample in adaptive training (default: 10)"
    echo "  --no_lora             Use full fine-tuning instead of LoRA"
    echo "  --no_adaptive         Disable adaptive training (use standard batch training)"
    echo "  --help                Show this help message"
    echo ""
    echo "Training modes:"
    echo "  - Adaptive (default): Train each sample until learned"
    echo "  - Standard: Fixed epochs with batch training (use --no_adaptive)"
    echo ""
    echo "Examples:"
    echo "  # Adaptive training with LoRA (default)"
    echo "  bash run_multiphase_pipeline.sh --model Qwen/Qwen2.5-0.5B-Instruct --train_samples 10000"
    echo ""
    echo "  # Standard batch training with full fine-tuning"
    echo "  bash run_multiphase_pipeline.sh --train_samples 10000 --no_lora --no_adaptive"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --train_samples)
            TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --test_samples)
            TEST_SAMPLES="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --inference_batch)
            INFERENCE_BATCH_SIZE="$2"
            shift 2
            ;;
        --train_batch)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --knowledge_epochs)
            KNOWLEDGE_EPOCHS="$2"
            shift 2
            ;;
        --num_trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS_PER_SAMPLE="$2"
            shift 2
            ;;
        --no_lora)
            NO_LORA="true"
            shift
            ;;
        --no_adaptive)
            ADAPTIVE="false"
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set learning rate based on training mode
if [ "$NO_LORA" = "true" ]; then
    LR="1e-5"
    TRAINING_MODE="Full Fine-tuning"
else
    LR="1e-4"
    TRAINING_MODE="LoRA"
fi

PROJECT_ROOT=$(dirname "$0")

# Set training mode string
if [ "$ADAPTIVE" = "true" ]; then
    ADAPTIVE_MODE="Adaptive (sample-by-sample)"
else
    ADAPTIVE_MODE="Standard (batch)"
fi

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
echo "Adaptive mode: $ADAPTIVE_MODE"
echo "Max steps per sample: $MAX_STEPS_PER_SAMPLE"
echo "Learning rate: $LR"
echo "Epochs (judgment): $EPOCHS"
echo "Epochs (knowledge): $KNOWLEDGE_EPOCHS"
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
    --lr $LR \
    --max_steps_per_sample $MAX_STEPS_PER_SAMPLE"

# Add --no_lora flag if full fine-tuning
if [ "$NO_LORA" = "true" ]; then
    CMD="$CMD --no_lora"
fi

# Add --adaptive flag if enabled
if [ "$ADAPTIVE" = "true" ]; then
    CMD="$CMD --adaptive"
fi

# Run the command
eval $CMD

echo ""
echo "=============================================="
echo "Multi-phase Pipeline completed!"
echo "=============================================="
echo "Results saved to: experiments/{auto-generated-name}/"
echo ""
echo "Training approach:"
echo "  - Adaptive: Each sample trained until learned (tested after each step)"
echo "  - Classification: 5/5=can, 0/5=cannot, 1-4/5=uncertain"
echo ""
echo "Expected improvements:"
echo "  Phase 1: Model learns to judge (trained until correct)"
echo "  Phase 2: Model learns knowledge (trained until answers correctly)"
echo "  Phase 3: Model updates judgment (becomes confident and accurate)"
echo "=============================================="
