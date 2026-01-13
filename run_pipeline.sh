#!/bin/bash
# Know-thyself: Full pipeline script
# Usage: bash run_pipeline.sh [model] [train_samples] [test_samples]

set -e  # Exit on error

# Default parameters
MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
TRAIN_SAMPLES=${2:-1000}
TEST_SAMPLES=${3:-100}
EPOCHS=3
BATCH_SIZE=4

echo "=============================================="
echo "Know-thyself Full Pipeline"
echo "=============================================="
echo "Model: $MODEL"
echo "Train samples: $TRAIN_SAMPLES"
echo "Test samples: $TEST_SAMPLES"
echo "=============================================="

# Step 1: Collect responses on train split
echo ""
echo "[Step 1/4] Collecting responses on train split..."
python run.py --step 1 \
    --model "$MODEL" \
    --num_samples "$TRAIN_SAMPLES" \
    --train_split train

# Step 2: Build training dataset
echo ""
echo "[Step 2/4] Building training dataset..."
python run.py --step 2

# Step 3: Train metacognition
echo ""
echo "[Step 3/4] Training metacognition model..."
python run.py --step 3 \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"

# Step 4: Evaluate on test split
echo ""
echo "[Step 4/4] Evaluating on test split..."
python run.py --step 4 \
    --model "$MODEL" \
    --test_samples "$TEST_SAMPLES" \
    --test_split test

echo ""
echo "=============================================="
echo "Pipeline completed!"
echo "=============================================="
