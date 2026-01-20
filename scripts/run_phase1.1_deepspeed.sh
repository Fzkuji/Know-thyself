#!/bin/bash
#
# Phase 1.1: Alternating Judgment and QA Training with DeepSpeed ZeRO-3
#
# This phase alternates between training judgment ability and QA ability:
# - Train judgment (can/uncertain/cannot) for 1 epoch
# - Train QA (learn to answer questions) for 1 epoch
# - Repeat for N epochs
#
# The idea is that as the model learns QA ability, its judgment about
# what it can answer should also improve, creating a virtuous cycle.
#
# Steps per cycle:
# 1. (Re-)Collect QA responses (to reflect current model ability)
# 2. Build judgment training data
# 3. Train judgment for 1 epoch
# 4. Evaluate judgment
# 5. Train QA for 1 epoch
# 6. Evaluate QA ability
# 7. Repeat
#

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ============== Parse Command Line Arguments ==============
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --val_samples)
            VAL_SAMPLES="$2"
            shift 2
            ;;
        --num_trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --train_batch_size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Phase 1.1: Alternating Judgment and QA Training with DeepSpeed ZeRO-3"
            echo ""
            echo "Options:"
            echo "  --model MODEL          Model name (default: Qwen/Qwen2.5-7B-Instruct)"
            echo "  --num_samples N        Training samples (default: 500)"
            echo "  --val_samples N        Validation samples (default: 1000)"
            echo "  --num_trials N         Trials per question (default: 10)"
            echo "  --num_epochs N         Number of alternating epochs (default: 10)"
            echo "  --batch_size N         Inference batch size (default: 16)"
            echo "  --train_batch_size N   Training batch size per GPU (default: 4)"
            echo "  --lr RATE              Learning rate (default: 1e-6)"
            echo "  --num_gpus N           Number of GPUs (default: 8)"
            echo "  --output_dir DIR       Output directory"
            echo "  --force                Force re-run all steps"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============== Configuration (with defaults) ==============
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
VAL_SAMPLES="${VAL_SAMPLES:-1000}"
FORCE="${FORCE:-0}"
NUM_TRIALS="${NUM_TRIALS:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
LR="${LR:-1e-6}"
NUM_GPUS="${NUM_GPUS:-8}"

# DeepSpeed config
DS_CONFIG="$PROJECT_DIR/configs/ds_config_zero3.json"

# Output directory
if [ -z "$OUTPUT_DIR" ]; then
    MODEL_SHORT=$(basename "$MODEL")
    OUTPUT_DIR="$PROJECT_DIR/experiments/phase1.1_ds_${MODEL_SHORT}"
fi
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "============== Configuration (Phase 1.1 DeepSpeed) =============="
echo "MODEL:       $MODEL"
echo "NUM_SAMPLES: $NUM_SAMPLES"
echo "VAL_SAMPLES: $VAL_SAMPLES"
echo "NUM_TRIALS:  $NUM_TRIALS"
echo "NUM_EPOCHS:  $NUM_EPOCHS (alternating judgment + QA)"
echo "BATCH_SIZE:  $BATCH_SIZE (inference)"
echo "TRAIN_BATCH: $TRAIN_BATCH_SIZE (per GPU)"
echo "LR:          $LR"
echo "NUM_GPUS:    $NUM_GPUS"
echo "OUTPUT_DIR:  $OUTPUT_DIR"
echo "DS_CONFIG:   $DS_CONFIG"
echo "FORCE:       $FORCE"
echo "=================================================================="

# ============== Helper Functions ==============
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Clean results file (only key metrics, no noise)
RESULTS_FILE="$OUTPUT_DIR/results.txt"

# Function to append epoch results to clean results file
append_results() {
    local epoch=$1
    local stage=$2  # "judgment" or "qa"
    local epoch_dir="$OUTPUT_DIR/epoch_${epoch}_${stage}"

    # Read metrics from JSON files
    python3 - "$epoch_dir" "$epoch" "$stage" >> "$RESULTS_FILE" << 'PYEOF'
import json
import sys
from pathlib import Path

epoch_dir = Path(sys.argv[1])
epoch = int(sys.argv[2])
stage = sys.argv[3]

train_json = epoch_dir / "metrics_train.json"
val_json = epoch_dir / "metrics_validation.json"

print(f"\n{'='*60}")
print(f"{'='*60}")
print(f"Epoch {epoch} - {stage.upper()} Results")
print(f"{'='*60}")
print(f"{'='*60}")

if train_json.exists():
    with open(train_json) as f:
        m = json.load(f)
    print(f"\n[TRAIN]")
    print(f"  QA Accuracy: {m.get('qa_accuracy', 0)*100:.1f}%")

    if stage == "judgment":
        print(f"  Judgment Accuracy: {m.get('exact_match_rate', 0)*100:.1f}%")

        # Confusion matrix
        c = m.get("confusion", {})
        print(f"\n  Confusion Matrix:")
        print(f"                        actual_can  actual_uncertain  actual_cannot")
        print(f"    predicted_can          {c.get('can_can', 0):5d}           {c.get('can_uncertain', 0):5d}            {c.get('can_cannot', 0):5d}")
        print(f"    predicted_uncertain    {c.get('uncertain_can', 0):5d}           {c.get('uncertain_uncertain', 0):5d}            {c.get('uncertain_cannot', 0):5d}")
        print(f"    predicted_cannot       {c.get('cannot_can', 0):5d}           {c.get('cannot_uncertain', 0):5d}            {c.get('cannot_cannot', 0):5d}")

        pred = m.get("pred_counts", {})
        actual = m.get("actual_counts", {})
        print(f"\n  Predicted: can={pred.get('can', 0)}, uncertain={pred.get('uncertain', 0)}, cannot={pred.get('cannot', 0)}")
        print(f"  Actual:    can={actual.get('can', 0)}, uncertain={actual.get('uncertain', 0)}, cannot={actual.get('cannot', 0)}")

if val_json.exists():
    with open(val_json) as f:
        m = json.load(f)
    print(f"\n[VALIDATION]")
    print(f"  QA Accuracy: {m.get('qa_accuracy', 0)*100:.1f}%")

    if stage == "judgment":
        print(f"  Judgment Accuracy: {m.get('exact_match_rate', 0)*100:.1f}%")

        # Confusion matrix
        c = m.get("confusion", {})
        print(f"\n  Confusion Matrix:")
        print(f"                        actual_can  actual_uncertain  actual_cannot")
        print(f"    predicted_can          {c.get('can_can', 0):5d}           {c.get('can_uncertain', 0):5d}            {c.get('can_cannot', 0):5d}")
        print(f"    predicted_uncertain    {c.get('uncertain_can', 0):5d}           {c.get('uncertain_uncertain', 0):5d}            {c.get('uncertain_cannot', 0):5d}")
        print(f"    predicted_cannot       {c.get('cannot_can', 0):5d}           {c.get('cannot_uncertain', 0):5d}            {c.get('cannot_cannot', 0):5d}")

        pred = m.get("pred_counts", {})
        actual = m.get("actual_counts", {})
        print(f"\n  Predicted: can={pred.get('can', 0)}, uncertain={pred.get('uncertain', 0)}, cannot={pred.get('cannot', 0)}")
        print(f"  Actual:    can={actual.get('can', 0)}, uncertain={actual.get('uncertain', 0)}, cannot={actual.get('cannot', 0)}")
PYEOF
}

# Initialize clean results file
echo "Phase 1.1 Alternating Training Results (DeepSpeed ZeRO-3)" > "$RESULTS_FILE"
echo "Model: $MODEL" >> "$RESULTS_FILE"
echo "Training samples: $NUM_SAMPLES, Validation samples: $VAL_SAMPLES" >> "$RESULTS_FILE"
echo "Num trials: $NUM_TRIALS, Learning rate: $LR" >> "$RESULTS_FILE"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_FILE"

# ============== Step 0: Baseline Evaluation ==============
log "Step 0: Baseline evaluation (before any training)"

mkdir -p "$OUTPUT_DIR/epoch_0_judgment"
if [ ! -f "$OUTPUT_DIR/baseline_eval.json" ] || [ "$FORCE" = "1" ]; then
    # Evaluate on training samples
    python "$SCRIPT_DIR/step4_evaluate.py" \
        --model "$MODEL" \
        --lora_path none \
        --split train \
        --num_samples "$NUM_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --epoch 0 \
        --output_json "$OUTPUT_DIR/epoch_0_judgment/metrics_train.json" \
        2>&1 | tee "$OUTPUT_DIR/baseline_train.log"

    # Evaluate on validation samples
    python "$SCRIPT_DIR/step4_evaluate.py" \
        --model "$MODEL" \
        --lora_path none \
        --split validation \
        --num_samples "$VAL_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --epoch 0 \
        --output_json "$OUTPUT_DIR/epoch_0_judgment/metrics_validation.json" \
        2>&1 | tee "$OUTPUT_DIR/baseline_val.log"

    echo '{"status": "completed"}' > "$OUTPUT_DIR/baseline_eval.json"
    log "Baseline evaluation completed"
fi

# Append baseline results to clean results file
append_results 0 "judgment"

# ============== Alternating Training Loop ==============
log "Starting alternating training loop ($NUM_EPOCHS epochs)"

# Track model path
CURRENT_MODEL="$MODEL"

for epoch in $(seq 1 $NUM_EPOCHS); do
    log "====== Epoch $epoch/$NUM_EPOCHS (Alternating: Judgment + QA) ======"

    # ============== Part A: Collect Responses & Train Judgment ==============
    log "--- Part A: Judgment Training ---"

    JUDGMENT_DIR="$OUTPUT_DIR/epoch_${epoch}_judgment"
    mkdir -p "$JUDGMENT_DIR"

    # Step 1: Collect QA Responses (using current model)
    RESPONSES_FILE="$JUDGMENT_DIR/responses.jsonl"
    if [ ! -f "$RESPONSES_FILE" ] || [ "$FORCE" = "1" ]; then
        log "Collecting QA responses with current model..."
        if python "$SCRIPT_DIR/step1_collect_responses.py" \
            --model "$CURRENT_MODEL" \
            --split train \
            --num_samples "$NUM_SAMPLES" \
            --num_trials "$NUM_TRIALS" \
            --inference_batch_size "$BATCH_SIZE" \
            --num_gpus "$NUM_GPUS" \
            --output "$RESPONSES_FILE" \
            2>&1 | tee "$JUDGMENT_DIR/step1.log"; then
            log "Response collection completed"
        else
            log "ERROR: Response collection failed!"
            exit 1
        fi
    else
        log "Responses already collected, skipping"
    fi

    # Step 2: Build Judgment Training Data
    JUDGMENT_DATA="$JUDGMENT_DIR/judgment_training_data.jsonl"
    if [ ! -f "$JUDGMENT_DATA" ] || [ "$FORCE" = "1" ]; then
        log "Building judgment training data..."
        if python "$SCRIPT_DIR/step2_build_dataset.py" \
            --input "$RESPONSES_FILE" \
            --output "$JUDGMENT_DATA" \
            2>&1 | tee "$JUDGMENT_DIR/step2.log"; then
            log "Judgment training data built"
        else
            log "ERROR: Judgment training data build failed!"
            exit 1
        fi
    else
        log "Judgment training data already exists, skipping"
    fi

    # Step 3: Train Judgment (1 epoch)
    JUDGMENT_MODEL="$JUDGMENT_DIR/model"
    mkdir -p "$JUDGMENT_MODEL"

    if [ ! -f "$JUDGMENT_MODEL/config.json" ] || [ "$FORCE" != "1" ]; then
        log "Training judgment epoch $epoch with DeepSpeed..."

        if deepspeed --num_gpus="$NUM_GPUS" \
            "$SCRIPT_DIR/step3_train_epoch_deepspeed.py" \
            --model "$CURRENT_MODEL" \
            --input "$JUDGMENT_DATA" \
            --output_dir "$JUDGMENT_MODEL" \
            --epoch "$epoch" \
            --lr "$LR" \
            --batch_size "$TRAIN_BATCH_SIZE" \
            --num_trials "$NUM_TRIALS" \
            --deepspeed "$DS_CONFIG" \
            2>&1 | tee "$JUDGMENT_DIR/train.log"; then

            # Verify model was saved
            if [ ! -f "$JUDGMENT_MODEL/config.json" ]; then
                log "ERROR: Training appeared to succeed but config.json not found!"
                exit 1
            fi
            log "Judgment epoch $epoch training complete"
        else
            log "ERROR: Judgment epoch $epoch training failed!"
            exit 1
        fi
    else
        log "Judgment model already trained, skipping"
    fi

    # Step 4: Evaluate Judgment
    log "Evaluating judgment epoch $epoch..."
    python "$SCRIPT_DIR/step4_evaluate.py" \
        --model "$JUDGMENT_MODEL" \
        --lora_path none \
        --split train \
        --num_samples "$NUM_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --epoch "$epoch" \
        --output_json "$JUDGMENT_DIR/metrics_train.json" \
        2>&1 | tee "$JUDGMENT_DIR/eval_train.log"

    python "$SCRIPT_DIR/step4_evaluate.py" \
        --model "$JUDGMENT_MODEL" \
        --lora_path none \
        --split validation \
        --num_samples "$VAL_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --epoch "$epoch" \
        --output_json "$JUDGMENT_DIR/metrics_validation.json" \
        2>&1 | tee "$JUDGMENT_DIR/eval_val.log"

    # Append judgment results
    append_results "$epoch" "judgment"

    # ============== Part B: Train QA ==============
    log "--- Part B: QA Training ---"

    QA_DIR="$OUTPUT_DIR/epoch_${epoch}_qa"
    mkdir -p "$QA_DIR"

    # Use the same responses.jsonl for QA training data
    # (responses.jsonl already has questions and answers)
    QA_DATA="$RESPONSES_FILE"

    # Step 5: Train QA (1 epoch)
    QA_MODEL="$QA_DIR/model"
    mkdir -p "$QA_MODEL"

    if [ ! -f "$QA_MODEL/config.json" ] || [ "$FORCE" != "1" ]; then
        log "Training QA epoch $epoch with DeepSpeed..."

        if deepspeed --num_gpus="$NUM_GPUS" \
            "$SCRIPT_DIR/step3_train_qa_epoch_deepspeed.py" \
            --model "$JUDGMENT_MODEL" \
            --input "$QA_DATA" \
            --output_dir "$QA_MODEL" \
            --epoch "$epoch" \
            --lr "$LR" \
            --batch_size "$TRAIN_BATCH_SIZE" \
            --deepspeed "$DS_CONFIG" \
            2>&1 | tee "$QA_DIR/train.log"; then

            # Verify model was saved
            if [ ! -f "$QA_MODEL/config.json" ]; then
                log "ERROR: QA training appeared to succeed but config.json not found!"
                exit 1
            fi
            # Update current model to the QA-trained model
            CURRENT_MODEL="$QA_MODEL"
            log "QA epoch $epoch training complete"
        else
            log "ERROR: QA epoch $epoch training failed!"
            exit 1
        fi
    else
        log "QA model already trained, skipping"
        CURRENT_MODEL="$QA_MODEL"
    fi

    # Step 6: Evaluate QA ability
    log "Evaluating QA epoch $epoch..."
    python "$SCRIPT_DIR/step4_evaluate.py" \
        --model "$QA_MODEL" \
        --lora_path none \
        --split train \
        --num_samples "$NUM_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --epoch "$epoch" \
        --output_json "$QA_DIR/metrics_train.json" \
        2>&1 | tee "$QA_DIR/eval_train.log"

    python "$SCRIPT_DIR/step4_evaluate.py" \
        --model "$QA_MODEL" \
        --lora_path none \
        --split validation \
        --num_samples "$VAL_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --epoch "$epoch" \
        --output_json "$QA_DIR/metrics_validation.json" \
        2>&1 | tee "$QA_DIR/eval_val.log"

    # Append QA results
    append_results "$epoch" "qa"

    log "Epoch $epoch complete (Judgment + QA)"
done

# ============== Final Summary ==============
log "Phase 1.1 (Alternating Training) complete!"
log "Final model: $CURRENT_MODEL"
log "Results in: $OUTPUT_DIR"
log "Clean results file: $RESULTS_FILE"

# Create symlink to final model
ln -sf "epoch_${NUM_EPOCHS}_qa/model" "$OUTPUT_DIR/final_model"
log "Symlink created: $OUTPUT_DIR/final_model -> epoch_${NUM_EPOCHS}_qa/model"

# Generate summary table
log "Generating summary table..."
python3 - "$OUTPUT_DIR" "$NUM_EPOCHS" << 'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

output_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/phase1.1"
num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

rows = []

# Baseline (epoch 0)
epoch_dir = Path(output_dir) / "epoch_0_judgment"
train_json = epoch_dir / "metrics_train.json"
val_json = epoch_dir / "metrics_validation.json"

row = {"epoch": 0, "stage": "baseline"}
if train_json.exists():
    with open(train_json) as f:
        m = json.load(f)
        row["train_qa"] = m.get("qa_accuracy", 0) * 100
        row["train_judgment"] = m.get("exact_match_rate", 0) * 100

if val_json.exists():
    with open(val_json) as f:
        m = json.load(f)
        row["val_qa"] = m.get("qa_accuracy", 0) * 100
        row["val_judgment"] = m.get("exact_match_rate", 0) * 100

if len(row) > 2:
    rows.append(row)

# Each epoch has judgment and qa
for epoch in range(1, num_epochs + 1):
    # Judgment
    epoch_dir = Path(output_dir) / f"epoch_{epoch}_judgment"
    train_json = epoch_dir / "metrics_train.json"
    val_json = epoch_dir / "metrics_validation.json"

    row = {"epoch": epoch, "stage": "judgment"}
    if train_json.exists():
        with open(train_json) as f:
            m = json.load(f)
            row["train_qa"] = m.get("qa_accuracy", 0) * 100
            row["train_judgment"] = m.get("exact_match_rate", 0) * 100

    if val_json.exists():
        with open(val_json) as f:
            m = json.load(f)
            row["val_qa"] = m.get("qa_accuracy", 0) * 100
            row["val_judgment"] = m.get("exact_match_rate", 0) * 100

    if len(row) > 2:
        rows.append(row)

    # QA
    epoch_dir = Path(output_dir) / f"epoch_{epoch}_qa"
    train_json = epoch_dir / "metrics_train.json"
    val_json = epoch_dir / "metrics_validation.json"

    row = {"epoch": epoch, "stage": "qa"}
    if train_json.exists():
        with open(train_json) as f:
            m = json.load(f)
            row["train_qa"] = m.get("qa_accuracy", 0) * 100
            row["train_judgment"] = m.get("exact_match_rate", 0) * 100

    if val_json.exists():
        with open(val_json) as f:
            m = json.load(f)
            row["val_qa"] = m.get("qa_accuracy", 0) * 100
            row["val_judgment"] = m.get("exact_match_rate", 0) * 100

    if len(row) > 2:
        rows.append(row)

print("\n" + "=" * 90)
print("PHASE 1.1 SUMMARY (Alternating Judgment + QA Training)")
print("=" * 90)
print(f"{'Epoch':>6} | {'Stage':>10} | {'Train QA':>10} | {'Train Judge':>12} | {'Val QA':>10} | {'Val Judge':>12}")
print("-" * 90)
for row in rows:
    epoch = row["epoch"]
    stage = row["stage"]
    train_qa = f"{row.get('train_qa', 0):.1f}%" if 'train_qa' in row else "N/A"
    train_j = f"{row.get('train_judgment', 0):.1f}%" if 'train_judgment' in row else "N/A"
    val_qa = f"{row.get('val_qa', 0):.1f}%" if 'val_qa' in row else "N/A"
    val_j = f"{row.get('val_judgment', 0):.1f}%" if 'val_judgment' in row else "N/A"
    print(f"{epoch:>6} | {stage:>10} | {train_qa:>10} | {train_j:>12} | {val_qa:>10} | {val_j:>12}")
print("=" * 90)

csv_path = Path(output_dir) / "summary.csv"
with open(csv_path, "w") as f:
    f.write("epoch,stage,train_qa_acc,train_judgment_acc,val_qa_acc,val_judgment_acc\n")
    for row in rows:
        f.write(f"{row['epoch']},{row['stage']},{row.get('train_qa', '')},{row.get('train_judgment', '')},{row.get('val_qa', '')},{row.get('val_judgment', '')}\n")
print(f"\nSummary saved to: {csv_path}")
PYTHON_SCRIPT

log "Done!"
