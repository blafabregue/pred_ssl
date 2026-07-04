#!/bin/bash
#SBATCH --job-name=pred_ssl_pretrain
#SBATCH --partition=<PARTITION>          # TODO: fill in your cluster partition
#SBATCH --account=<ACCOUNT>              # TODO: fill in your account (or remove)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                        # TODO: adjust
#SBATCH --time=24:00:00                  # cluster wall-clock cap (resume across jobs)
#SBATCH --output=pred_ssl/logs/%x_%j.out
#SBATCH --error=pred_ssl/logs/%x_%j.err
#
# Pretrain ONE experiment (framework/experiment/seed), AUTO-RESUMING from its last
# checkpoint. Built for a wall-clock-limited cluster: if the job is killed at the time
# limit, just resubmit the SAME command — it continues where training stopped. Normally
# submitted by scripts/slurm_submit.sh; the checkpoint bookkeeping lives here.
set -e

FRAMEWORK=${FRAMEWORK:-simclr}
EXPERIMENT=${EXPERIMENT:-relpred}
ARCH=${ARCH:-resnet50}
SEED=${SEED:-1}
EPOCHS=${EPOCHS:-500}
SAVE_FREQ=${SAVE_FREQ:-25}               # resume granularity (epochs lost on a kill <= this)
IN100=${IN100:-./pred_ssl/datasets/imagenet100}
SAVE_DIR=${SAVE_DIR:-./pred_ssl/checkpoints/${FRAMEWORK}_${EXPERIMENT}_${ARCH}_s${SEED}}
LOG=${LOG:-./pred_ssl/logs/${FRAMEWORK}_${EXPERIMENT}_${ARCH}_s${SEED}.log}
CONFIG_OVERLAY=${CONFIG_OVERLAY:-}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"     # repo root (parent of pred_ssl/)
mkdir -p "$(dirname "$LOG")" "${SAVE_DIR}"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-pytorch_2_0_0}" || echo "WARN: could not activate ${CONDA_ENV}"
fi

FINAL="${SAVE_DIR}/checkpoint_$(printf '%04d' "${EPOCHS}").pth.tar"
if [ -f "${FINAL}" ]; then
    echo "already complete: ${FINAL} exists — nothing to do" | tee -a "${LOG}"
    exit 0
fi

# auto-resume: prefer the rolling checkpoint_last.pth.tar, else the highest milestone
RESUME=""
if [ -f "${SAVE_DIR}/checkpoint_last.pth.tar" ]; then
    RESUME="--resume ${SAVE_DIR}/checkpoint_last.pth.tar"
else
    latest=$(ls -1 "${SAVE_DIR}"/checkpoint_[0-9]*.pth.tar 2>/dev/null | sort | tail -1 || true)
    [ -n "${latest}" ] && RESUME="--resume ${latest}"
fi

{
    echo "=========================================="
    echo "pretrain ${FRAMEWORK}/${EXPERIMENT} seed=${SEED} arch=${ARCH} (job ${SLURM_JOB_ID:-local})"
    echo "resume: ${RESUME:-<fresh start>}   started: $(date)"
    echo "=========================================="
} | tee -a "${LOG}"

# append (never truncate) so the log keeps history across resubmits
python -m pred_ssl.train --framework "${FRAMEWORK}" --experiment "${EXPERIMENT}" \
    --arch "${ARCH}" --data "${IN100}" --epochs "${EPOCHS}" --seed "${SEED}" \
    --save-dir "${SAVE_DIR}" --save-latest --save-freq "${SAVE_FREQ}" \
    ${CONFIG_OVERLAY:+--config-overlay "${CONFIG_OVERLAY}"} ${RESUME} 2>&1 | tee -a "${LOG}"

echo "pretrain finished through epoch ${EPOCHS}: $(date)" | tee -a "${LOG}"
