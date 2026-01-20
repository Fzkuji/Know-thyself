#!/bin/bash
# Unified Training Script for Know-thyself
#
# Usage:
#   ./scripts/run_training.sh                    # Default: judgment training, binary mode, batch training
#   ./scripts/run_training.sh --mode knowledge   # Knowledge training
#   ./scripts/run_training.sh --label_mode uncertainty --training_mode adaptive  # Custom settings

set -e

# Default settings
MODE=${MODE:-"judgment"}              # judgment or knowledge
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
OUTPUT_DIR=${OUTPUT_DIR:-"experiments/training"}
EPOCHS=${EPOCHS:-10}
LR=${LR:-"1e-5"}
BATCH_SIZE=${BATCH_SIZE:-4}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-1}
NUM_GPUS=${NUM_GPUS:-8}

# Judgment-specific settings
LABEL_MODE=${LABEL_MODE:-"binary"}          # binary or uncertainty
TRAINING_MODE=${TRAINING_MODE:-"batch"}     # batch or adaptive
NUM_TRIALS=${NUM_TRIALS:-10}                # for uncertainty mode
TEMPERATURE=${TEMPERATURE:-0.7}             # for uncertainty mode

# Knowledge-specific settings
FILTER_ABILITY=${FILTER_ABILITY:-""}        # e.g., "cannot uncertain"

# DeepSpeed config
DS_CONFIG=${DS_CONFIG:-"configs/ds_config_zero3.json"}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation)
            GRADIENT_ACCUMULATION="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --label_mode)
            LABEL_MODE="$2"
            shift 2
            ;;
        --training_mode)
            TRAINING_MODE="$2"
            shift 2
            ;;
        --num_trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --filter_ability)
            FILTER_ABILITY="$2"
            shift 2
            ;;
        --ds_config)
            DS_CONFIG="$2"
            shift 2
            ;;
        --input)
            INPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Change to project directory
cd "$(dirname "$0")/.."

echo "========================================"
echo "Know-thyself Training"
echo "========================================"
echo "Mode: $MODE"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Num GPUs: $NUM_GPUS"
echo "DeepSpeed config: $DS_CONFIG"

if [ "$MODE" == "judgment" ]; then
    echo "Label mode: $LABEL_MODE"
    echo "Training mode: $TRAINING_MODE"
    if [ "$LABEL_MODE" == "uncertainty" ]; then
        echo "Num trials: $NUM_TRIALS"
        echo "Temperature: $TEMPERATURE"
    fi
elif [ "$MODE" == "knowledge" ]; then
    if [ -n "$FILTER_ABILITY" ]; then
        echo "Filter ability: $FILTER_ABILITY"
    fi
fi
echo "========================================"

# Run training
if [ "$MODE" == "judgment" ]; then
    # Build command
    CMD="deepspeed --num_gpus=$NUM_GPUS scripts/train_judgment_deepspeed.py"
    CMD="$CMD --model $MODEL"
    CMD="$CMD --output_dir $OUTPUT_DIR"
    CMD="$CMD --epochs $EPOCHS"
    CMD="$CMD --lr $LR"
    CMD="$CMD --batch_size $BATCH_SIZE"
    CMD="$CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION"
    CMD="$CMD --label_mode $LABEL_MODE"
    CMD="$CMD --training_mode $TRAINING_MODE"
    CMD="$CMD --deepspeed $DS_CONFIG"

    if [ -n "$INPUT" ]; then
        CMD="$CMD --input $INPUT"
    else
        CMD="$CMD --num_samples $NUM_SAMPLES"
    fi

    if [ "$LABEL_MODE" == "uncertainty" ]; then
        CMD="$CMD --num_trials $NUM_TRIALS"
        CMD="$CMD --temperature $TEMPERATURE"
    fi

    echo "Running: $CMD"
    eval $CMD

elif [ "$MODE" == "knowledge" ]; then
    # Build command
    CMD="deepspeed --num_gpus=$NUM_GPUS scripts/train_knowledge_deepspeed.py"
    CMD="$CMD --model $MODEL"
    CMD="$CMD --output_dir $OUTPUT_DIR"
    CMD="$CMD --epochs $EPOCHS"
    CMD="$CMD --lr $LR"
    CMD="$CMD --batch_size $BATCH_SIZE"
    CMD="$CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION"
    CMD="$CMD --deepspeed $DS_CONFIG"

    if [ -n "$INPUT" ]; then
        CMD="$CMD --input $INPUT"
    else
        CMD="$CMD --num_samples $NUM_SAMPLES"
    fi

    if [ -n "$FILTER_ABILITY" ]; then
        CMD="$CMD --filter_ability $FILTER_ABILITY"
    fi

    echo "Running: $CMD"
    eval $CMD

else
    echo "Unknown mode: $MODE"
    echo "Use --mode judgment or --mode knowledge"
    exit 1
fi

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
