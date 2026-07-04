#!/bin/bash
#SBATCH --job-name=pred_ssl_pretrain
#SBATCH --partition=<PARTITION>          # TODO: fill in your cluster partition
#SBATCH --account=<ACCOUNT>              # TODO: fill in your account (or remove)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                        # TODO: adjust
#SBATCH --time=48:00:00                  # TODO: adjust
#SBATCH --output=pred_ssl/logs/%x_%j.out
#SBATCH --error=pred_ssl/logs/%x_%j.err
#
# Pretrain one (framework, experiment). Submit with env vars, e.g.:
#   sbatch --export=ALL,FRAMEWORK=simclr,EXPERIMENT=relpred,ARCH=resnet50,EPOCHS=500 \
#          pred_ssl/scripts/sbatch_pretrain.slurm
set -e

FRAMEWORK=${FRAMEWORK:-simclr}
EXPERIMENT=${EXPERIMENT:-relpred}
ARCH=${ARCH:-resnet50}
EPOCHS=${EPOCHS:-500}
IN100=${IN100:-./pred_ssl/datasets/imagenet100}

# repo root = parent of the pred_ssl/ package
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_2_0_0}"

MODE=pretrain FRAMEWORK="${FRAMEWORK}" EXPERIMENT="${EXPERIMENT}" ARCH="${ARCH}" \
    EPOCHS="${EPOCHS}" IN100="${IN100}" bash pred_ssl/scripts/run_pipeline.sh
