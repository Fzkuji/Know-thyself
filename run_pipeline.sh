#!/bin/bash
# Know-thyself: Full pipeline script with before/after comparison
# Usage: bash run_pipeline.sh [model] [train_samples] [test_samples] [inference_batch_size]

set -e  # Exit on error

# Default parameters
MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
TRAIN_SAMPLES=${2:-1000}
TEST_SAMPLES=${3:-100}
INFERENCE_BATCH_SIZE=${4:-16}
EPOCHS=3
BATCH_SIZE=4

echo "=============================================="
echo "Know-thyself Full Pipeline"
echo "=============================================="
echo "Model: $MODEL"
echo "Train samples: $TRAIN_SAMPLES"
echo "Test samples: $TEST_SAMPLES"
echo "Inference batch size: $INFERENCE_BATCH_SIZE"
echo "=============================================="

# Step 0: Baseline evaluation (BEFORE training)
echo ""
echo "[Step 0/5] Baseline evaluation (before training)..."
python run.py --step 0 \
    --model "$MODEL" \
    --test_samples "$TEST_SAMPLES" \
    --test_split test \
    --inference_batch_size "$INFERENCE_BATCH_SIZE"

# Step 1: Collect responses on train split
echo ""
echo "[Step 1/5] Collecting responses on train split..."
python run.py --step 1 \
    --model "$MODEL" \
    --num_samples "$TRAIN_SAMPLES" \
    --train_split train \
    --inference_batch_size "$INFERENCE_BATCH_SIZE"

# Step 2: Build training dataset
echo ""
echo "[Step 2/5] Building training dataset..."
python run.py --step 2

# Step 3: Train metacognition
echo ""
echo "[Step 3/5] Training metacognition model..."
python run.py --step 3 \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"

# Step 4: Evaluate on test split (AFTER training)
echo ""
echo "[Step 4/5] Evaluating on test split (after training)..."
python run.py --step 4 \
    --model "$MODEL" \
    --test_samples "$TEST_SAMPLES" \
    --test_split test \
    --inference_batch_size "$INFERENCE_BATCH_SIZE"

echo ""
echo "=============================================="
echo "Pipeline completed!"
echo "Compare Step 0 (baseline) vs Step 4 (trained)"
echo "=============================================="
