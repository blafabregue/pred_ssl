#!/bin/bash
#SBATCH --job-name=pred_ssl_eval
#SBATCH --partition=<PARTITION>          # TODO: fill in your cluster partition
#SBATCH --account=<ACCOUNT>              # TODO: fill in your account (or remove)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                        # TODO: adjust
#SBATCH --time=24:00:00
#SBATCH --output=pred_ssl/logs/%x_%j.out
#SBATCH --error=pred_ssl/logs/%x_%j.err
#
# Evaluate ONE finished pretraining (its final checkpoint): IN-100 linear, IN-100
# rotation, CUB-200 linear, Flowers-102 few-shot -> one <tag>.eval.log ending in the
# EVAL_DONE marker that slurm_status.py / slurm_submit.sh look for. Submitted by
# scripts/slurm_submit.sh once pretraining is complete.
set -e

TAG=${TAG:-simclr_relpred_resnet50_s1}
ARCH=${ARCH:-resnet50}
EPOCHS=${EPOCHS:-500}
EVAL_EPOCHS=${EVAL_EPOCHS:-200}
SAVE_DIR=${SAVE_DIR:-./pred_ssl/checkpoints/${TAG}}
EVAL_LOG=${EVAL_LOG:-./pred_ssl/logs/${TAG}.eval.log}
IN100=${IN100:-./pred_ssl/datasets/imagenet100}
CUB=${CUB:-./pred_ssl/datasets/cub200_prepared}
FLOWERS=${FLOWERS:-./pred_ssl/datasets/flowers102_prepared}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p "$(dirname "$EVAL_LOG")"
# environment: scripts/env.sh (cluster-specific: module load / conda / venv), else conda
if [ -f pred_ssl/scripts/env.sh ]; then
    source pred_ssl/scripts/env.sh
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-pytorch_2_0_0}" || echo "WARN: could not activate ${CONDA_ENV}"
fi

CKPT="${SAVE_DIR}/checkpoint_$(printf '%04d' "${EPOCHS}").pth.tar"
[ -f "${CKPT}" ] || { echo "ERROR: final checkpoint ${CKPT} not found (pretrain not done)"; exit 1; }

: > "${EVAL_LOG}"
echo "eval ${TAG}  ckpt=${CKPT}  started: $(date)" | tee -a "${EVAL_LOG}"

python -m pred_ssl.eval.linear_probe --data "${IN100}" --pretrained "${CKPT}" \
    --arch "${ARCH}" --epochs "${EVAL_EPOCHS}" 2>&1 | tee -a "${EVAL_LOG}"
python -m pred_ssl.eval.linear_probe --data "${IN100}" --pretrained "${CKPT}" \
    --arch "${ARCH}" --epochs "${EVAL_EPOCHS}" --eval-rotation 2>&1 | tee -a "${EVAL_LOG}"
python -m pred_ssl.eval.linear_probe --data "${CUB}" --pretrained "${CKPT}" \
    --arch "${ARCH}" --epochs "${EVAL_EPOCHS}" 2>&1 | tee -a "${EVAL_LOG}"
python -m pred_ssl.eval.few_shot --data "${FLOWERS}" --pretrained "${CKPT}" \
    --arch "${ARCH}" 2>&1 | tee -a "${EVAL_LOG}"

echo "EVAL_DONE $(date)" | tee -a "${EVAL_LOG}"
