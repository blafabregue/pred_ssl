#!/bin/bash
#SBATCH --job-name=pred_ssl_eval
#SBATCH --partition=<PARTITION>          # TODO: fill in your cluster partition
#SBATCH --account=<ACCOUNT>              # TODO: fill in your account (or remove)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                        # TODO: adjust
#SBATCH --time=24:00:00                  # TODO: adjust
#SBATCH --output=pred_ssl/logs/%x_%j.out
#SBATCH --error=pred_ssl/logs/%x_%j.err
#
# Run the four evals (IN-100 linear, IN-100 rotation, CUB-200 linear, Flowers
# few-shot) against a pretrained checkpoint. Submit with env vars, e.g.:
#   sbatch --export=ALL,FRAMEWORK=simclr,EXPERIMENT=relpred,ARCH=resnet50,EPOCHS=500 \
#          pred_ssl/scripts/sbatch_eval.slurm
# (EPOCHS selects checkpoint_<EPOCHS>.pth.tar; or pass CKPT=<path> explicitly.)
set -e

FRAMEWORK=${FRAMEWORK:-simclr}
EXPERIMENT=${EXPERIMENT:-relpred}
ARCH=${ARCH:-resnet50}
EPOCHS=${EPOCHS:-500}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_2_0_0}"

MODE=eval FRAMEWORK="${FRAMEWORK}" EXPERIMENT="${EXPERIMENT}" ARCH="${ARCH}" \
    EPOCHS="${EPOCHS}" ${CKPT:+CKPT="${CKPT}"} bash pred_ssl/scripts/run_pipeline.sh
