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

# Default parameters (always full fine-tuning, no LoRA)
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_SAMPLES=1000
TEST_SAMPLES=100
DATASET="triviaqa"
INFERENCE_BATCH_SIZE=16
TRAIN_BATCH_SIZE=32
EPOCHS=10
KNOWLEDGE_EPOCHS=10
NUM_TRIALS=5
ADAPTIVE="true"
MAX_STEPS_PER_SAMPLE=10
LR="1e-5"  # Learning rate for full fine-tuning
FORCE="false"
EXPERIMENT=""
PHASE=""
SUMMARY="false"
NUM_GPUS=""
DDP="false"

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
    echo "  --epochs              Epochs for judgment training (default: 10)"
    echo "  --knowledge_epochs    Epochs for knowledge training (default: 10)"
    echo "  --num_trials          Responses per question (default: 5)"
    echo "  --max_steps           Max steps per sample in adaptive training (default: 10)"
    echo "  --lr                  Learning rate (default: 1e-5 for full fine-tuning)"
    echo "  --no_adaptive         Disable adaptive training (use standard batch training)"
    echo "  --experiment          Experiment name (to resume or re-run specific experiment)"
    echo "  --phase               Run specific phase only (1, 2, or 3)"
    echo "  --force               Force re-run even if phase already completed"
    echo "  --summary             Print summary of existing experiment (requires --experiment)"
    echo "  --num_gpus N          Number of GPUs to use for inference (default: all available)"
    echo "  --ddp                 Use DDP for multi-GPU training (gradient sync across GPUs)"
    echo "  --help                Show this help message"
    echo ""
    echo "Training modes (always full fine-tuning):"
    echo "  - Single GPU training + Multi-GPU inference (default)"
    echo "  - DDP training + Multi-GPU inference (use --ddp)"
    echo ""
    echo "Fine-tuning modes:"
    echo "  - Adaptive (default): Train each sample until learned"
    echo "  - Standard: Fixed epochs with batch training (use --no_adaptive)"
    echo ""
    echo "Examples:"
    echo "  # Adaptive training with full fine-tuning (default)"
    echo "  bash run_multiphase_pipeline.sh --model Qwen/Qwen2.5-0.5B-Instruct --train_samples 10000"
    echo ""
    echo "  # Standard batch training with full fine-tuning"
    echo "  bash run_multiphase_pipeline.sh --train_samples 10000 --no_adaptive"
    echo ""
    echo "  # DDP training with full fine-tuning (multi-GPU gradient sync)"
    echo "  bash run_multiphase_pipeline.sh --train_samples 10000 --ddp"
    echo ""
    echo "  # Print summary of existing experiment"
    echo "  bash run_multiphase_pipeline.sh --summary --experiment Qwen2.5-7B_triviaqa_train1000_test100_0115_1430"
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
        --lr)
            LR="$2"
            shift 2
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
        --force)
            FORCE="true"
            shift
            ;;
        --summary)
            SUMMARY="true"
            shift
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --ddp)
            DDP="true"
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

# Always use full fine-tuning
TRAINING_MODE="Full Fine-tuning"

PROJECT_ROOT=$(dirname "$0")

# Handle --summary mode: just print existing experiment summary
if [ "$SUMMARY" = "true" ]; then
    if [ -z "$EXPERIMENT" ]; then
        echo "Error: --summary requires --experiment <name>"
        echo "Usage: bash run_multiphase_pipeline.sh --summary --experiment <experiment_name>"
        exit 1
    fi
    python $PROJECT_ROOT/scripts/run_multiphase.py --summary --experiment "$EXPERIMENT"
    exit 0
fi

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
if [ -n "$EXPERIMENT" ]; then
    echo "Experiment: $EXPERIMENT"
else
    echo "(Experiment name will be auto-generated)"
fi
if [ -n "$PHASE" ]; then
    echo "Phase: $PHASE only"
fi
if [ "$FORCE" = "true" ]; then
    echo "Force mode: ON (will re-run completed phases)"
fi
if [ -n "$NUM_GPUS" ]; then
    echo "Number of GPUs: $NUM_GPUS"
else
    echo "Number of GPUs: all available (auto-detect)"
fi
if [ "$DDP" = "true" ]; then
    echo "DDP Training: ON (multi-GPU gradient sync)"
else
    echo "DDP Training: OFF (single-GPU training + multi-GPU inference)"
fi
echo "=============================================="

# Build command based on DDP mode
if [ "$DDP" = "true" ]; then
    # Determine number of GPUs for torchrun
    if [ -n "$NUM_GPUS" ]; then
        NPROC=$NUM_GPUS
    else
        # Auto-detect available GPUs
        NPROC=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
    fi
    echo "Launching with torchrun (nproc_per_node=$NPROC)..."
    CMD="torchrun --nproc_per_node=$NPROC $PROJECT_ROOT/scripts/run_multiphase_ddp.py \
        --ddp \
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
else
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
fi

# Add --adaptive flag if enabled
if [ "$ADAPTIVE" = "true" ]; then
    CMD="$CMD --adaptive"
fi

# Add --experiment if specified
if [ -n "$EXPERIMENT" ]; then
    CMD="$CMD --experiment $EXPERIMENT"
fi

# Add --phase if specified
if [ -n "$PHASE" ]; then
    CMD="$CMD --phase $PHASE"
fi

# Add --force if specified
if [ "$FORCE" = "true" ]; then
    CMD="$CMD --force"
fi

# Add --num_gpus if specified
if [ -n "$NUM_GPUS" ]; then
    CMD="$CMD --num_gpus $NUM_GPUS"
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
echo "  - Epoch-level filtering: Before each epoch, test all samples"
echo "  - Only train samples that are still incorrect (skip already learned)"
echo "  - Classification: 5/5=can, 0/5=cannot, 1-4/5=uncertain"
echo ""
echo "Expected improvements:"
echo "  Phase 1: Model learns to judge (trained until correct)"
echo "  Phase 2: Model learns knowledge (trained until answers correctly)"
echo "  Phase 3: Model updates judgment (becomes confident and accurate)"
echo "=============================================="
