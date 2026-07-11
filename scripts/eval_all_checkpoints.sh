#!/bin/bash
# Evaluate all checkpoints (10-130) on 4 GPUs in parallel.
# Each GPU runs one eval at a time, sequentially processing its assigned checkpoints.

set -e

REPO=/mnt/nvme0n1/rawhad/self_distillation/aligning_lm_from_user_interaction
PYTHON="$REPO/.venv_vllm/bin/python"
SCRIPT="$REPO/scripts/eval_maas_sdft.py"
TEST_JSONL=/home/lab/rawhad/sdg-ki-eval/data/maas_data/rohans_data/test_maas_sdft.jsonl
CKPT_DIR="$REPO/output/sdft-offpolicy-rohans_data-run-1"
OUT_DIR="$REPO/eval_results/sdft-offpolicy-rohans_data-run-1"

mkdir -p "$OUT_DIR"

# Copy base model results
cp "$REPO/eval_results/base_model/eval_results.jsonl" "$OUT_DIR/base_model.jsonl" 2>/dev/null || true

# 13 checkpoints across 4 GPUs
# GPU 0: 10 20 30 40
# GPU 1: 50 60 70 80
# GPU 2: 90 100 110
# GPU 3: 120 130

run_gpu() {
    local gpu=$1
    shift
    for step in "$@"; do
        echo "[GPU $gpu] Evaluating checkpoint-$step ..."
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON $SCRIPT \
            --model "$CKPT_DIR/checkpoint-$step" \
            --test_jsonl "$TEST_JSONL" \
            --output_dir "$OUT_DIR/tmp_ckpt_$step" \
            2>&1 | tee "$OUT_DIR/ckpt-${step}.log"

        # Move result to final location
        mv "$OUT_DIR/tmp_ckpt_$step/eval_results.jsonl" "$OUT_DIR/ckpt-${step}.jsonl"
        rm -rf "$OUT_DIR/tmp_ckpt_$step"
        echo "[GPU $gpu] Done checkpoint-$step"
    done
}

run_gpu 0 10 20 30 40 &
run_gpu 1 50 60 70 80 &
run_gpu 2 90 100 110 &
run_gpu 3 120 130 &

wait
echo "All evaluations complete. Results in $OUT_DIR/"
