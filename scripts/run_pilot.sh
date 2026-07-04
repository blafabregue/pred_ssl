#!/bin/bash
# Phase-2 pilot: short ResNet-18 relpred run on an ImageNet-100 subset, then the
# automated gate check. Non-SLURM (matches the existing nohup/CUDA_VISIBLE_DEVICES
# convention); run on the cluster GPU.
#
# Usage:
#   GPU=0 bash pred_ssl/scripts/run_pilot.sh
#   GPU=0 EPOCHS=20 FRAMEWORK=simclr bash pred_ssl/scripts/run_pilot.sh
#
# Defaults match run.sh and the relctl pilot knobs (50 epochs, 20 classes x 500).
# Anything much shorter leaves the relational head at chance (~50%) and fails the
# gate's ">= 3 factors learning" check for lack of gradient steps, not for a bug.
set -e

# --- config (override via env) ---
GPU=${GPU:-0}
FRAMEWORK=${FRAMEWORK:-simclr}
EPOCHS=${EPOCHS:-50}
ARCH=${ARCH:-resnet18}
SRC=${SRC:-./pred_ssl/datasets/imagenet100} # full IN-100 dataset root
SUBSET=${SUBSET:-./pred_ssl/pilot_in100}    # symlinked pilot subset
N_CLASSES=${N_CLASSES:-20}
N_PER_CLASS=${N_PER_CLASS:-500}
SAVE_DIR=${SAVE_DIR:-./pred_ssl/checkpoints/pilot_${FRAMEWORK}}
LOG=${LOG:-./pred_ssl/logs/pilot_${FRAMEWORK}.log}
CONDA_ENV=${CONDA_ENV:-pytorch_2_0_0}
CONFIG_OVERLAY=${CONFIG_OVERLAY:-}        # optional YAML overlay for YAML-only knobs (relctl)

export CUDA_VISIBLE_DEVICES=${GPU}
# cd to repo root (parent of pred_ssl/)
cd "$(dirname "$0")/../.."
mkdir -p "$(dirname "$LOG")"

# conda env (override with CONDA_ENV=...)
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}" || echo "WARN: could not activate ${CONDA_ENV} (continuing in current env)"
fi

echo "=========================================="
echo "Phase-2 pilot: ${FRAMEWORK} / ${ARCH} / ${EPOCHS} epochs"
echo "Started: $(date)"
echo "=========================================="

# Build the symlinked pilot subset (idempotent: existing symlinks are skipped, so a
# partial subset from an interrupted run is simply completed).
python pred_ssl/scripts/make_pilot_subset.py --src "${SRC}" --dst "${SUBSET}" \
    --n-classes "${N_CLASSES}" --n-per-class "${N_PER_CLASS}" --splits train val

# Pilot pretraining (tee to log + console, matching the repo convention).
python -m pred_ssl.train --framework "${FRAMEWORK}" --experiment relpred \
    --arch "${ARCH}" --data "${SUBSET}" --epochs "${EPOCHS}" \
    --batch-size 256 --workers 8 --save-dir "${SAVE_DIR}" \
    ${CONFIG_OVERLAY:+--config-overlay "${CONFIG_OVERLAY}"} 2>&1 | tee "${LOG}"

echo "=========================================="
echo "Pilot gate check"
echo "=========================================="
python pred_ssl/scripts/check_pilot_gate.py "${LOG}"   # exit code gates scaling
echo "Finished: $(date)"
