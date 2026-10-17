#!/usr/bin/env bash
# ==============================================================================
# PE-CIR: End-to-End Clean Pipeline from Scratch (B1-clean -> B4-new -> B5-new)
# ==============================================================================
# All models are trained with identical seed: 42 on the reference-disjoint clean split.
# Scientific GO Gate enforces: 95% Cluster-Robust Bootstrap CI (10,000 iters) > 0.
# ==============================================================================

set -euo pipefail

IMAGE_ROOT="${1:-/home/iec/datasets/cc3m/subset_5000/images}"
OUTPUT_DIR="${2:-runs/clean_pipeline_from_scratch}"
TRAIN_MANIFEST="${3:-}"
SEED=42
BATCH_SIZE=16

# Auto-detect train manifest if not specified
if [ -z "${TRAIN_MANIFEST}" ]; then
    if [ -f "/home/iec/datasets/cc3m/pseudo_screening/train.csv" ]; then
        TRAIN_MANIFEST="/home/iec/datasets/cc3m/pseudo_screening/train.csv"
    elif [ -f "data/train.csv" ]; then
        TRAIN_MANIFEST="data/train.csv"
    elif [ -f "runs/controlled_repair_experiment/clean_data/clean_train.csv" ]; then
        TRAIN_MANIFEST="runs/controlled_repair_experiment/clean_data/clean_train.csv"
    elif [ -f "output/BAN_GIAO_CHO_CO_20261010/1_train_manifest/train.csv" ]; then
        TRAIN_MANIFEST="output/BAN_GIAO_CHO_CO_20261010/1_train_manifest/train.csv"
    else
        echo "[ERROR] Could not find training manifest train.csv!"
        echo "Please provide path as 3rd argument: $0 <image_root> <output_dir> <train_manifest>"
        exit 1
    fi
fi

# Fallback image root if default not found
if [ ! -d "${IMAGE_ROOT}" ] && [ -d "data/images" ]; then
    IMAGE_ROOT="data/images"
fi

echo "=============================================================================="
echo "PE-CIR: MASTER CLEAN EXPERIMENT PIPELINE (SEED: ${SEED})"
echo "=============================================================================="
echo "Train manifest: ${TRAIN_MANIFEST}"
echo "Image root    : ${IMAGE_ROOT}"
echo "Output dir    : ${OUTPUT_DIR}"
echo "Python binary : $(which python)"
echo "Start time    : $(date)"
echo "=============================================================================="

mkdir -p "${OUTPUT_DIR}"

# Step 1: Run End-to-End Controlled Repair Pipeline (Train B1 -> Train B4 -> GO Gate -> Train B5)
echo ""
echo ">>> STEP 1/2: Running Controlled Training Pipeline (B1 -> B4 -> Cluster CI Gate -> B5)..."
python scripts/controlled_repair_b4_to_b5.py \
    --train-manifest "${TRAIN_MANIFEST}" \
    --image-root "${IMAGE_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --train-b1 \
    --b1-epochs 3 \
    --b4-epochs 3 \
    --b5-epochs 3 \
    --batch-size ${BATCH_SIZE} \
    --seed ${SEED} \
    --cf-margin 0.15 \
    --lambda-cf 0.8 \
    --n-boot 10000 \
    --device auto

# Step 2: Run Full Multi-Metric Comparison (B1 vs B4) with 10,000 Cluster Bootstrap Replicates
echo ""
echo ">>> STEP 2/2: Running Full Triplet + Gallery Comparison with 10,000 Cluster Bootstrap iterations..."
python scripts/compare_b1_b4_clean_val.py \
    --manifest "${OUTPUT_DIR}/clean_data/clean_val.csv" \
    --image-root "${IMAGE_ROOT}" \
    --b1-checkpoint "${OUTPUT_DIR}/B1_clean_train/best_checkpoint.pt" \
    --b4-checkpoint "${OUTPUT_DIR}/B4_controlled_repair/best_checkpoint.pt" \
    --output-dir "${OUTPUT_DIR}/comparison_b1_b4" \
    --n-boot 10000 \
    --device auto

echo ""
echo "=============================================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "Protocol file  : ${OUTPUT_DIR}/controlled_repair_protocol.json"
echo "GO Gate report : ${OUTPUT_DIR}/b4_go_gate_report.json"
echo "Comparison JSON: ${OUTPUT_DIR}/comparison_b1_b4/comparison_metrics.json"
echo "End time       : $(date)"
echo "=============================================================================="
