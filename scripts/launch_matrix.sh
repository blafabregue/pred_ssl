#!/bin/bash
# Submit the full matrix: 4 frameworks x {baseline, relpred} (optionally +
# relpred_lambda0). For each combo: sbatch pretrain, then sbatch eval with an
# afterok dependency on the pretrain job. Edit the #SBATCH placeholders in
# sbatch_pretrain.slurm / sbatch_eval.slurm before running.
#
# Usage:
#   ARCH=resnet50 EPOCHS=500 bash pred_ssl/scripts/launch_matrix.sh
#   INCLUDE_ABLATION=1 bash pred_ssl/scripts/launch_matrix.sh    # adds relpred_lambda0
set -e

ARCH=${ARCH:-resnet50}
EPOCHS=${EPOCHS:-500}
FRAMEWORKS=${FRAMEWORKS:-"simclr moco byol looc vicreg"}
EXPERIMENTS=${EXPERIMENTS:-"baseline relpred"}
[ "${INCLUDE_ABLATION:-0}" = "1" ] && EXPERIMENTS="${EXPERIMENTS} relpred_lambda0"

cd "$(dirname "$0")/../.."
mkdir -p pred_ssl/logs

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found. Run this on the SLURM cluster." >&2
    exit 1
fi

for fw in ${FRAMEWORKS}; do
    for exp in ${EXPERIMENTS}; do
        echo "=== submitting ${fw} / ${exp} (${ARCH}, ${EPOCHS} ep) ==="
        pre=$(sbatch --parsable \
            --job-name="pre_${fw}_${exp}" \
            --export=ALL,FRAMEWORK=${fw},EXPERIMENT=${exp},ARCH=${ARCH},EPOCHS=${EPOCHS} \
            pred_ssl/scripts/sbatch_pretrain.slurm)
        echo "  pretrain job: ${pre}"
        evl=$(sbatch --parsable --dependency=afterok:${pre} \
            --job-name="eval_${fw}_${exp}" \
            --export=ALL,FRAMEWORK=${fw},EXPERIMENT=${exp},ARCH=${ARCH},EPOCHS=${EPOCHS} \
            pred_ssl/scripts/sbatch_eval.slurm)
        echo "  eval job:     ${evl} (after ${pre})"
    done
done

echo
echo "Submitted. After all jobs finish, collect results with:"
echo "  python pred_ssl/scripts/extract_results.py --logs-dir ./pred_ssl/logs --out ./pred_ssl/results.csv"
