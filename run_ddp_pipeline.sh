#!/bin/bash
# Know-thyself: DDP Multi-phase Pipeline
# Uses DistributedDataParallel for multi-GPU training
#
# Each GPU processes different samples, gradients are synchronized,
# then parameters are updated together. Effective batch_size = num_gpus.
#
# Usage: bash run_ddp_pipeline.sh --model Qwen/Qwen2.5-7B-Instruct --train_samples 1000

set -e

# Default parameters
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_SAMPLES=1000
TEST_SAMPLES=100
DATASET="triviaqa"
INFERENCE_BATCH_SIZE=16
TRAIN_BATCH_SIZE=32
NO_LORA="false"
EPOCHS=10
KNOWLEDGE_EPOCHS=10
NUM_TRIALS=5
ADAPTIVE="true"
MAX_STEPS_PER_SAMPLE=10
FORCE="false"
EXPERIMENT=""
PHASE=""
NUM_GPUS=""

# Help function
show_help() {
    echo "Usage: bash run_ddp_pipeline.sh [OPTIONS]"
    echo ""
    echo "DDP Training: Each GPU processes 1 sample, gradients are synchronized."
    echo "Effective batch size = number of GPUs."
    echo ""
    echo "Options:"
    echo "  --model               Model name (default: Qwen/Qwen2.5-0.5B-Instruct)"
    echo "  --train_samples       Number of training samples (default: 1000)"
    echo "  --test_samples        Number of test samples (default: 100)"
    echo "  --dataset             Dataset name (default: triviaqa)"
    echo "  --inference_batch     Batch size for inference (default: 16)"
    echo "  --epochs              Epochs for judgment training (default: 10)"
    echo "  --knowledge_epochs    Epochs for knowledge training (default: 10)"
    echo "  --num_trials          Responses per question (default: 5)"
    echo "  --max_steps           Max steps per sample (default: 10)"
    echo "  --no_lora             Use full fine-tuning instead of LoRA"
    echo "  --no_adaptive         Disable adaptive training"
    echo "  --experiment          Experiment name (to resume)"
    echo "  --phase               Run specific phase only (1, 2, or 3)"
    echo "  --num_gpus N          Number of GPUs to use (default: all available)"
    echo "  --force               Force re-run even if phase completed"
    echo "  --help                Show this help message"
    echo ""
    echo "Example:"
    echo "  bash run_ddp_pipeline.sh --model Qwen/Qwen2.5-7B-Instruct --train_samples 1024 --num_gpus 8"
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
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --force)
            FORCE="true"
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

# Detect number of GPUs
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
fi

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
echo "Know-thyself DDP Multi-phase Pipeline"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Train samples: $TRAIN_SAMPLES"
echo "Test samples: $TEST_SAMPLES"
echo "Training mode: $TRAINING_MODE"
echo "Adaptive mode: $ADAPTIVE"
echo "Learning rate: $LR"
echo "Epochs (judgment): $EPOCHS"
echo "Epochs (knowledge): $KNOWLEDGE_EPOCHS"
echo ""
echo "DDP Configuration:"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Effective batch size: $NUM_GPUS (1 sample per GPU)"
echo "=============================================="

# Build command
CMD="torchrun --nproc_per_node=$NUM_GPUS $PROJECT_ROOT/scripts/run_multiphase_ddp.py \
    --model $MODEL \
    --dataset $DATASET \
    --num_samples $TRAIN_SAMPLES \
    --test_samples $TEST_SAMPLES \
    --num_trials $NUM_TRIALS \
    --inference_batch_size $INFERENCE_BATCH_SIZE \
    --epochs $EPOCHS \
    --knowledge_epochs $KNOWLEDGE_EPOCHS \
    --lr $LR \
    --max_steps_per_sample $MAX_STEPS_PER_SAMPLE \
    --ddp"

# Add flags
if [ "$NO_LORA" = "true" ]; then
    CMD="$CMD --no_lora"
fi

if [ "$ADAPTIVE" = "true" ]; then
    CMD="$CMD --adaptive"
fi

if [ -n "$EXPERIMENT" ]; then
    CMD="$CMD --experiment $EXPERIMENT"
fi

if [ -n "$PHASE" ]; then
    CMD="$CMD --phase $PHASE"
fi

if [ "$FORCE" = "true" ]; then
    CMD="$CMD --force"
fi

# Run
echo "Running: $CMD"
eval $CMD

echo ""
echo "=============================================="
echo "DDP Pipeline completed!"
echo "=============================================="
