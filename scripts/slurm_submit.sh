#!/bin/bash
# Submit every UNFINISHED experiment in the matrix (scripts/experiments.py), one SLURM
# job each. Pretraining AUTO-RESUMES from its last checkpoint, so a job killed at the
# 24h wall-clock cap is simply continued the next time you run this. Re-run after each
# batch expires until `slurm_status` reports everything DONE.
#
#   bash pred_ssl/scripts/slurm_submit.sh
#   FRAMEWORKS="simclr moco" VARIANTS="baseline relpred" SEEDS="1 2 3" \
#       bash pred_ssl/scripts/slurm_submit.sh          # narrow the matrix
#
# Fill the <PARTITION>/<ACCOUNT> placeholders in sbatch_pretrain.slurm + sbatch_eval.slurm
# first. Per experiment: submit pretrain if not finished, else submit eval if not done.
set -e

cd "$(dirname "$0")/../.."          # repo root (parent of pred_ssl/)
mkdir -p pred_ssl/logs

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found — run on the SLURM cluster." >&2; exit 1; }

IN100=${IN100:-./pred_ssl/datasets/imagenet100}
CUB=${CUB:-./pred_ssl/datasets/cub200_prepared}
FLOWERS=${FLOWERS:-./pred_ssl/datasets/flowers102_prepared}
EVAL_EPOCHS=${EVAL_EPOCHS:-200}

# names of jobs already queued/running, to avoid double-submitting
QUEUED="$(squeue --me --noheader --format=%j 2>/dev/null || true)"
is_queued() { printf '%s\n' "${QUEUED}" | grep -qx "$1"; }

n_pre=0; n_eval=0; n_skip=0
while IFS=$'\t' read -r tag framework experiment arch seed epochs save_dir log; do
    final="${save_dir}/checkpoint_$(printf '%04d' "${epochs}").pth.tar"
    eval_log="./pred_ssl/logs/${tag}.eval.log"

    if [ ! -f "${final}" ]; then                       # pretrain not finished
        if is_queued "pre_${tag}"; then echo "queued   pre_${tag}"; n_skip=$((n_skip+1)); continue; fi
        jid=$(sbatch --parsable --job-name="pre_${tag}" \
            --export=ALL,FRAMEWORK=${framework},EXPERIMENT=${experiment},ARCH=${arch},SEED=${seed},EPOCHS=${epochs},SAVE_DIR=${save_dir},LOG=${log},IN100=${IN100} \
            pred_ssl/scripts/sbatch_pretrain.slurm)
        echo "submit   pre_${tag}  -> ${jid}"; n_pre=$((n_pre+1)); continue
    fi

    if grep -q "EVAL_DONE" "${eval_log}" 2>/dev/null; then echo "done     ${tag}"; n_skip=$((n_skip+1)); continue; fi
    if is_queued "eval_${tag}"; then echo "queued   eval_${tag}"; n_skip=$((n_skip+1)); continue; fi
    jid=$(sbatch --parsable --job-name="eval_${tag}" \
        --export=ALL,TAG=${tag},ARCH=${arch},EPOCHS=${epochs},SAVE_DIR=${save_dir},EVAL_LOG=${eval_log},IN100=${IN100},CUB=${CUB},FLOWERS=${FLOWERS},EVAL_EPOCHS=${EVAL_EPOCHS} \
        pred_ssl/scripts/sbatch_eval.slurm)
    echo "submit   eval_${tag}  -> ${jid}"; n_eval=$((n_eval+1))
done < <(python -m pred_ssl.scripts.experiments --format tsv)

echo
echo "submitted ${n_pre} pretrain + ${n_eval} eval job(s); skipped ${n_skip} (done/queued)."
echo "watch progress:  python -m pred_ssl.scripts.slurm_status"
