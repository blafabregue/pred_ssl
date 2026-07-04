#!/bin/bash
# =============================================================================
# pred_ssl CONTROL PANEL  —  one entrypoint to configure, PREVIEW, and run.
#
#   1. Edit the CONFIG block below (or override any value with an env var).
#   2. Preview exactly what will happen, WITHOUT running anything:
#         bash pred_ssl/run.sh --dry-run
#   3. Run it (asks for confirmation first, unless --yes):
#         bash pred_ssl/run.sh
#
# MODES (set MODE= below or pass --mode X):
#   test      run the CPU unit-test suite only (no GPU, no data)         -> sanity that the code is intact
#   pilot     short ResNet-18 run on a small IN-100 subset + gate check  -> "is the signal learnable?"
#   pipeline  full: pretrain -> IN-100 lincls -> rotation -> CUB -> Flowers, per framework/experiment
#   matrix    submit the whole grid (FRAMEWORKS x EXPERIMENTS) via SLURM
#
# Every CONFIG value can also be set as an env var, e.g.:
#   MODE=pilot GPU=1 EPOCHS=50 bash pred_ssl/run.sh --dry-run
# =============================================================================
set -uo pipefail

# ============================== CONFIG (edit me) =============================
MODE="${MODE:-pilot}"                         # test | pilot | pipeline | matrix
FRAMEWORKS="${FRAMEWORKS:-simclr}"            # any of: simclr moco byol looc vicreg
EXPERIMENTS="${EXPERIMENTS:-relpred}"         # any of: baseline relpred relpred_lambda0 relpred_decoupled relpred_proj3 relpred_split
ARCH="${ARCH:-resnet18}"                      # resnet18 | resnet50
EPOCHS="${EPOCHS:-50}"                         # pretraining epochs
EVAL_EPOCHS="${EVAL_EPOCHS:-200}"             # linear-probe epochs (pipeline mode)
GPU="${GPU:-0}"                                # CUDA device index
CONDA_ENV="${CONDA_ENV:-pytorch_2_0_0}"       # conda env to activate

# Data (paths are relative to the repo root)
IN100="${IN100:-./pred_ssl/datasets/imagenet100}"
CUB="${CUB:-./pred_ssl/datasets/cub200_prepared}"
FLOWERS="${FLOWERS:-./pred_ssl/datasets/flowers102_prepared}"

# Pilot-only knobs
PILOT_CLASSES="${PILOT_CLASSES:-20}"
PILOT_PER_CLASS="${PILOT_PER_CLASS:-500}"
# ============================================================================

DRY_RUN=0
ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --yes|-y) ASSUME_YES=1 ;;
        --mode=*) MODE="${arg#*=}" ;;
        --mode) ;;                       # handled below
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) ;;
    esac
done
# allow "--mode X" (space form)
prev=""; for arg in "$@"; do [ "$prev" = "--mode" ] && MODE="$arg"; prev="$arg"; done

cd "$(dirname "$0")/.."                    # repo root (parent of pred_ssl/)
REPO_ROOT="$(pwd)"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  [ OK ] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }

# --------------------------- prerequisite checks ----------------------------
CHECK_FAILED=0
preflight() {
    bold "Prerequisite checks"
    # conda env (informational only — the python-deps import below is the real gate).
    # Read $CONDA_DEFAULT_ENV directly; piping `conda env list` into grep can make
    # conda crash with a spurious BrokenPipeError.
    if [ "${CONDA_DEFAULT_ENV:-}" = "${CONDA_ENV}" ]; then
        ok "conda env '${CONDA_ENV}' is active"
    elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
        warn "active conda env is '${CONDA_DEFAULT_ENV}', not '${CONDA_ENV}' (relying on the python-deps check)"
    else
        warn "no active conda env detected (relying on the python-deps check)"
    fi
    # python deps
    if python -c "import torch, torchvision, numpy, PIL, yaml" 2>/dev/null; then
        ok "python deps present (torch, torchvision, numpy, PIL, yaml)"
    else
        bad "missing python deps — need torch torchvision numpy pillow pyyaml"; CHECK_FAILED=1
    fi
    # GPU (not needed for 'test')
    if [ "$MODE" != "test" ]; then
        if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            ok "CUDA available (GPU ${GPU})"
        else
            warn "torch.cuda.is_available() is False — will run on CPU (slow)"
        fi
    fi
    # data
    case "$MODE" in
        pilot|pipeline|matrix)
            if [ -d "${IN100}/train" ]; then
                ok "ImageNet-100 found (${IN100})"
            else
                bad "ImageNet-100 not found at ${IN100}/train"; CHECK_FAILED=1
            fi ;;
    esac
    if [ "$MODE" = "pipeline" ] || [ "$MODE" = "matrix" ]; then
        [ -d "${CUB}/train" ]     && ok "CUB-200 found (${CUB})"        || warn "CUB-200 missing (${CUB}) — STEP 4 will fail; prepare it or skip"
        [ -d "${FLOWERS}/train" ] && ok "Flowers-102 found (${FLOWERS})" || warn "Flowers-102 missing (${FLOWERS}) — STEP 5 will fail; prepare it or skip"
    fi
    if [ "$MODE" = "matrix" ]; then
        command -v sbatch >/dev/null 2>&1 && ok "sbatch available" || { bad "sbatch not found (matrix mode needs SLURM)"; CHECK_FAILED=1; }
        grep -q "<PARTITION>" pred_ssl/scripts/sbatch_pretrain.slurm 2>/dev/null \
            && warn "edit #SBATCH placeholders in pred_ssl/scripts/sbatch_*.slurm first" \
            || ok "SLURM #SBATCH placeholders filled in"
    fi
}

# --------------------------- the execution plan -----------------------------
declare -a CMDS
build_plan() {
    CMDS=()
    case "$MODE" in
        test)
            CMDS+=("python -m pytest pred_ssl/tests/ -q") ;;
        pilot)
            CMDS+=("GPU=${GPU} CONDA_ENV=${CONDA_ENV} ARCH=${ARCH} EPOCHS=${EPOCHS} \
SRC=${IN100} N_CLASSES=${PILOT_CLASSES} N_PER_CLASS=${PILOT_PER_CLASS} \
FRAMEWORK=$(echo "$FRAMEWORKS" | awk '{print $1}') bash pred_ssl/scripts/run_pilot.sh") ;;
        pipeline)
            for fw in $FRAMEWORKS; do for exp in $EXPERIMENTS; do
                CMDS+=("GPU=${GPU} CONDA_ENV=${CONDA_ENV} FRAMEWORK=${fw} EXPERIMENT=${exp} \
ARCH=${ARCH} EPOCHS=${EPOCHS} EVAL_EPOCHS=${EVAL_EPOCHS} \
IN100=${IN100} CUB=${CUB} FLOWERS=${FLOWERS} bash pred_ssl/scripts/run_pipeline.sh")
            done; done ;;
        matrix)
            CMDS+=("ARCH=${ARCH} EPOCHS=${EPOCHS} CONDA_ENV=${CONDA_ENV} \
FRAMEWORKS=\"${FRAMEWORKS}\" EXPERIMENTS=\"${EXPERIMENTS}\" bash pred_ssl/scripts/launch_matrix.sh")
            CMDS+=("# after all jobs finish: python pred_ssl/scripts/extract_results.py --logs-dir ./pred_ssl/logs --out ./pred_ssl/results.csv") ;;
        *)
            echo "Unknown MODE='$MODE' (use: test | pilot | pipeline | matrix)"; exit 2 ;;
    esac
}

print_plan() {
    bold "Resolved configuration"
    cat <<EOF
  repo root     : ${REPO_ROOT}
  MODE          : ${MODE}
  frameworks    : ${FRAMEWORKS}
  experiments   : ${EXPERIMENTS}
  arch / epochs : ${ARCH} / ${EPOCHS}     (eval epochs: ${EVAL_EPOCHS})
  GPU / env     : ${GPU} / ${CONDA_ENV}
  data          : IN100=${IN100}  CUB=${CUB}  FLOWERS=${FLOWERS}
EOF
    [ "$MODE" = "pilot" ] && echo "  pilot subset  : ${PILOT_CLASSES} classes x ${PILOT_PER_CLASS} imgs"
    echo "  outputs       : checkpoints in pred_ssl/checkpoints/, logs in pred_ssl/logs/"
    echo
    bold "Will run ${#CMDS[@]} command(s):"
    local i=1
    for c in "${CMDS[@]}"; do
        if [[ "$c" == \#* ]]; then echo "      ${c}"; else echo "  [$i] ${c}"; i=$((i+1)); fi
    done
}

# --------------------------------- main -------------------------------------
echo "============================================================"
bold "pred_ssl control panel"
echo "============================================================"
preflight
echo
build_plan
print_plan
echo "------------------------------------------------------------"

if [ "$DRY_RUN" -eq 1 ]; then
    bold "DRY RUN — nothing was executed."
    [ "$CHECK_FAILED" -eq 1 ] && echo "(note: some prerequisite checks failed — fix them before a real run)"
    exit 0
fi

if [ "$CHECK_FAILED" -eq 1 ]; then
    echo "ERROR: prerequisite checks failed (see [FAIL] above). Aborting."
    echo "       Re-run with --dry-run to inspect, or fix and retry."
    exit 1
fi

if [ "$ASSUME_YES" -ne 1 ]; then
    read -r -p "Proceed with the above? [y/N] " ans
    case "$ans" in [yY]|[yY][eE][sS]) ;; *) echo "Aborted."; exit 0 ;; esac
fi

for c in "${CMDS[@]}"; do
    [[ "$c" == \#* ]] && continue
    echo; bold ">>> $c"
    eval "$c"
done
echo; bold "Done."
