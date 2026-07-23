#!/bin/bash
#SBATCH --job-name=pred_ssl_pretrain
#SBATCH --partition=publicgpu          # TODO: fill in your cluster partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                        # TODO: adjust
#SBATCH --time=24:00:00                  # cluster wall-clock cap (resume across jobs)
#SBATCH --output=pred_ssl/logs/%x_%j.out
#SBATCH --error=pred_ssl/logs/%x_%j.err
#SBATCH --constraint="gpuh100|gpua100|gpuh200|gpua40|gpul40s"
#
# Pretrain ONE experiment (framework/experiment/seed), AUTO-RESUMING from its last
# checkpoint. Built for a wall-clock-limited cluster: if the job is killed at the time
# limit, just resubmit the SAME command — it continues where training stopped. Normally
# submitted by scripts/slurm_submit.sh; the checkpoint bookkeeping lives here.
# pipefail: without it, `python ... | tee` reports tee's exit code (0), so a crashed
# run would still print "pretrain finished" and look complete. With pipefail + set -e
# a training crash aborts the job (non-zero), as it should.
set -eo pipefail

FRAMEWORK=${FRAMEWORK:-simclr}
EXPERIMENT=${EXPERIMENT:-relpred}
ARCH=${ARCH:-resnet50}
SEED=${SEED:-1}
EPOCHS=${EPOCHS:-500}
SAVE_FREQ=${SAVE_FREQ:-10}               # checkpoint_last cadence (epochs lost on a kill <= this)
IN100=${IN100:-./pred_ssl/datasets/imagenet100}
SAVE_DIR=${SAVE_DIR:-./pred_ssl/checkpoints/${FRAMEWORK}_${EXPERIMENT}_${ARCH}_s${SEED}}
LOG=${LOG:-./pred_ssl/logs/${FRAMEWORK}_${EXPERIMENT}_${ARCH}_s${SEED}.log}
CONFIG_OVERLAY=${CONFIG_OVERLAY:-}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"     # repo root (parent of pred_ssl/)
mkdir -p "$(dirname "$LOG")" "${SAVE_DIR}"
# environment: scripts/env.sh (cluster-specific: module load / conda / venv), else conda
if [ -f pred_ssl/scripts/env.sh ]; then
    source pred_ssl/scripts/env.sh
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-pytorch_2_0_0}" || echo "WARN: could not activate ${CONDA_ENV}"
fi

# Reduce allocator fragmentation (VICReg's 8192-d expander + two ResNet-50 graphs run
# close to the card's limit; the OOM message itself recommends this).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Refuse to run on CPU: a node whose GPU faulted (e.g. ECC error) can leave CUDA
# unavailable, and training would silently fall back to CPU at ~400s/iter — hours of
# wasted wall-clock. Fail fast instead so the job can be requeued onto a healthy node.
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA not available on this node — refusing to run on CPU. Requeue." \
        | tee -a "${LOG}"
    exit 1
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
