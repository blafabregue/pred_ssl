#!/bin/bash
# One (framework, experiment) end-to-end: pretrain -> IN-100 lincls -> IN-100
# rotation -> CUB-200 lincls -> Flowers-102 few-shot, chained into ONE log with
# STEP markers so scripts/extract_results.py can parse it. Non-SLURM (nohup /
# CUDA_VISIBLE_DEVICES convention); also reused as the body of the sbatch wrappers.
#
# Usage:
#   GPU=0 FRAMEWORK=simclr EXPERIMENT=relpred bash pred_ssl/scripts/run_pipeline.sh
#   MODE=pretrain ... bash run_pipeline.sh    # pretrain only
#   MODE=eval     ... bash run_pipeline.sh    # evals only (needs an existing checkpoint)
set -e

GPU=${GPU:-0}
FRAMEWORK=${FRAMEWORK:-simclr}
EXPERIMENT=${EXPERIMENT:-relpred}
ARCH=${ARCH:-resnet50}
EPOCHS=${EPOCHS:-500}
EVAL_EPOCHS=${EVAL_EPOCHS:-200}
MODE=${MODE:-all}                      # all | pretrain | eval

# Paths are relative to the repo root (this script cd's there below).
IN100=${IN100:-./pred_ssl/datasets/imagenet100}
CUB=${CUB:-./pred_ssl/datasets/cub200_prepared}
FLOWERS=${FLOWERS:-./pred_ssl/datasets/flowers102_prepared}

CONDA_ENV=${CONDA_ENV:-pytorch_2_0_0}
CONFIG_OVERLAY=${CONFIG_OVERLAY:-}        # optional YAML overlay for YAML-only knobs (relctl)
TAG=${FRAMEWORK}_${EXPERIMENT}
SAVE_DIR=${SAVE_DIR:-./pred_ssl/checkpoints/${TAG}}
CKPT=${CKPT:-${SAVE_DIR}/checkpoint_$(printf '%04d' "${EPOCHS}").pth.tar}
LOG=${LOG:-./pred_ssl/logs/${TAG}.log}

export CUDA_VISIBLE_DEVICES=${GPU}
cd "$(dirname "$0")/../.."          # repo root (parent of pred_ssl/)
mkdir -p "$(dirname "$LOG")" "${SAVE_DIR}"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}" || echo "WARN: could not activate ${CONDA_ENV} (continuing in current env)"
fi

log() { echo "$@" | tee -a "${LOG}"; }

: > "${LOG}"   # truncate
log "=========================================="
log "pred_ssl pipeline: ${TAG} (${ARCH}, mode=${MODE})"
log "Started: $(date)"
log "=========================================="

if [ "${MODE}" = "all" ] || [ "${MODE}" = "pretrain" ]; then
    log "STEP 1: Pretrain (${EPOCHS} epochs)"
    python -m pred_ssl.train --framework "${FRAMEWORK}" --experiment "${EXPERIMENT}" \
        --arch "${ARCH}" --data "${IN100}" --epochs "${EPOCHS}" \
        --save-dir "${SAVE_DIR}" \
        ${CONFIG_OVERLAY:+--config-overlay "${CONFIG_OVERLAY}"} 2>&1 | tee -a "${LOG}"
fi

if [ "${MODE}" = "all" ] || [ "${MODE}" = "eval" ]; then
    log "STEP 2: ImageNet-100 Object Linear Eval"
    python -m pred_ssl.eval.linear_probe --data "${IN100}" --pretrained "${CKPT}" \
        --arch "${ARCH}" --epochs "${EVAL_EPOCHS}" 2>&1 | tee -a "${LOG}"

    log "STEP 3: ImageNet-100 Rotation Linear Eval"
    python -m pred_ssl.eval.linear_probe --data "${IN100}" --pretrained "${CKPT}" \
        --arch "${ARCH}" --epochs "${EVAL_EPOCHS}" --eval-rotation 2>&1 | tee -a "${LOG}"

    log "STEP 4: CUB-200 Object Linear Eval"
    python -m pred_ssl.eval.linear_probe --data "${CUB}" --pretrained "${CKPT}" \
        --arch "${ARCH}" --epochs "${EVAL_EPOCHS}" 2>&1 | tee -a "${LOG}"

    log "STEP 5: Flowers-102 Few-shot Eval"
    python -m pred_ssl.eval.few_shot --data "${FLOWERS}" --pretrained "${CKPT}" \
        --arch "${ARCH}" 2>&1 | tee -a "${LOG}"
fi

log "=========================================="
log "DONE: ${TAG}   Finished: $(date)"
log "=========================================="
