# pred_ssl — Relational / Pairwise Augmentation-Prediction SSL

A unified, config-driven SSL repo implementing a **relational** auxiliary task: a
3-layer head looks at *both* views' post-avgpool backbone features `(h1, h2)` and
predicts, **per augmentation factor**, whether the *same* parameter value was applied
to both views (binary same/different). Framework-agnostic across SimCLR / MoCo / BYOL
/ LooC / VICReg. Total loss `L = L_ssl + λ · mean_f BCE_f`, `λ = 0.5`.

This is a standalone sibling of the existing `*-Imagenet*` repos; it does not modify
any of them. It reuses their conventions (ImageNet-100 / CUB-200 / Flowers-102 data
layout, `pytorch_2_0_0` conda env, "Best Val" logging, checkpoint format).

## Layout
```
configs/    base.yaml + framework/{simclr,moco,byol,looc,vicreg}.yaml + experiment/{baseline,relpred,relpred_lambda0,relpred_decoupled}.yaml
data/       transforms.py (parameterized per-factor sharing + labels + mask), loader.py
models/     backbones.py (resnet + avgpool hook), rel_head.py, frameworks/{simclr,moco,byol,looc,vicreg}.py
losses.py   NTXentLoss, BYOLLoss, RelPairLoss (per-factor BCE)
train.py    single entrypoint (--framework / --experiment)
eval/       linear_probe.py (IN-100, CUB-200, --eval-rotation), few_shot.py (Flowers-102)
scripts/    run_pipeline.sh, run_pilot.sh, make_pilot_subset.py, check_pilot_gate.py,
            sbatch_pretrain.slurm, sbatch_eval.slurm, launch_matrix.sh, extract_results.py
tests/      pytest suite (transforms, models, frameworks, pilot gate, extract)
```

## The 9 factors
`rotation, hflip, brightness, contrast, saturation, hue, grayscale, blur, crop`.
Per factor: with prob `p_same=0.5` the identical parameter is applied to both views
(label "same"), else a guaranteed-different value (discrete: exclude view-1's value;
continuous: ≥δ gap; crop: "same" = identical box, "different" = boxes with IoU ≤
`delta.crop`). Saturation/hue are masked out of the loss whenever either view is
grayscale. NOTE: sharing the crop means that, in the coupled `relpred` experiment,
about half the contrastive positives see the SAME crop box, which weakens the
crop-invariance signal — `relpred_decoupled` avoids this (the SSL pair keeps fully
independent crops there).

## Projection head (all frameworks)
`proj_preset=native` (default) keeps each framework's original projection head
byte-for-byte, so reference numbers are reproduced exactly. `proj_preset=custom` swaps
in a shared, fully-sized MLP for every framework: `proj_layers` (number of Linear
layers), `proj_hidden` (hidden width), `proj_out` (output dim) and `proj_bn` (BatchNorm
between layers). `proj_out` also drives the coupled dims (MoCo/LooC queue width, BYOL
predictor). VICReg's `native` head is its own configurable expander (`vicreg_*`).

## Experiments
- `baseline`   — standard independent augmentation, no head (`rel_lambda=0`). Reproduces existing numbers.
- `relpred`    — the method (sharing loader + head, `rel_lambda=0.5`).
- `relpred_lambda0` — ablation: sharing loader, head off. (Not a pure-SSL control.)
- `relpred_decoupled` — the method WITHOUT the augmentation confound: the contrastive
  loss sees a standard independent pair, and a separate per-factor shared/different
  pair feeds only the head (`rel_decoupled=true`). Costs 2 extra backbone forwards/step.
- `relpred_proj3` — `relpred` plus the new custom 3-layer projection head
  (`proj_preset=custom`, `proj_layers=3`).
- `relpred_split` — `relpred` plus the **latent split** (disentanglement): h is
  partitioned into `[vanilla | common | rel]` (ratios `split_ratios`, default
  0.5/0.25/0.25); the SSL head consumes vanilla+common, the relational head
  common+rel, so each exclusive block only receives its own loss's gradient.
  Two ratio variants probe the partition geometry: `relpred_split_80_10_10`
  (vanilla-heavy) and `relpred_split_45_45_10` (common-heavy, tiny exclusive
  blocks).
  Optional `split_decov_lambda > 0` actively decorrelates the exclusive blocks.
  Measure it with `eval.linear_probe --feat-slice vanilla|common|rel` (rotation
  probe should be strong on `rel`, object probe strong on `vanilla`).

For running the full study on a 24h-limited SLURM cluster (one job per experiment,
multiple seeds, auto-resume + a status report), see **HANDOFF.md §7b**.

## Interactive control panel — `relctl` (recommended)

`relctl` is a full interactive TUI for driving everything from one place over SSH —
edit **any** knob, run **any** action, and manage long jobs, without hand-editing
files or remembering flags.

```bash
python -m pred_ssl.relctl            # auto: Rich tier if installed, else plain
python -m pred_ssl.relctl --plain    # force the zero-dependency plain tier
python -m pred_ssl.relctl --validate # check the knob catalog vs configs/, then exit
```

- **Configure** — 7 grouped editors expose every knob with live validation: globals,
  the active framework's block (e.g. SimCLR `temperature`, MoCo `moco_k`, BYOL
  `tau_base`), optimizer/LR (with a live `base_lr` readout), augmentation, the
  relational head (incl. the per-factor `delta` dict), eval/probe hyperparameters,
  and runtime/paths/pilot.
- **Run** — pilot · pipeline · pretrain-only · eval-only · single eval step · matrix
  (SLURM) · resume · make-subset · gate-check · extract-results · tests. The launch
  screen shows the **resolved** config, the exact command(s), the generated YAML
  overlay, and `run.sh`'s preflight checklist before you commit.
- **Jobs** — launches long runs in the background (`nohup`, or `tmux` if present;
  `sbatch` for matrix) and returns immediately. List/tail/stop, with live epoch +
  per-factor progress scraped from the logs. The registry survives SSH drops.
- **Profiles** — save/load named configurations (tracked in `relctl/profiles/`).

Zero hard dependencies beyond what the `pytorch_2_0_0` env already has (stdlib +
pyyaml). `pip install --user rich` upgrades the rendering automatically. Edited
YAML-only knobs travel to `train.py` via the new `--config-overlay` flag, so the
committed `configs/*.yaml` are never mutated. `python -m pred_ssl.train --print-config`
prints the fully-resolved merged config (what `relctl`'s **verify** uses).

## Static control panel — `pred_ssl/run.sh` (scriptable / non-interactive)

A single bash script to configure, **preview, and run** everything — handy for
scripts and SLURM. Edit the CONFIG block at the top (or override any value with an
env var), preview with `--dry-run` (prints the prerequisite checklist, the resolved
config, and the exact commands — runs nothing), then run it (it asks for confirmation
first, unless `--yes`). `relctl` reuses this script's preflight + dry-run.

```bash
# 1) see what WILL happen, without running anything:
bash pred_ssl/run.sh --dry-run

# 2) run it (confirm prompt):       3) or pick a mode / override knobs:
bash pred_ssl/run.sh                  MODE=pilot GPU=1 EPOCHS=50 bash pred_ssl/run.sh --dry-run
```

Modes: `test` (CPU unit tests only) · `pilot` (short ResNet-18 run + automated gate) ·
`pipeline` (full pretrain→4 evals per framework/experiment) · `matrix` (full SLURM grid).
A real run aborts if any prerequisite check fails; `--dry-run` only reports them.

## Quick start (cluster, `pytorch_2_0_0`)
```bash
# Phase-2 sanity pilot + automated gate (resnet18, IN-100 subset)
GPU=0 bash pred_ssl/scripts/run_pilot.sh

# One full pipeline (pretrain -> IN-100 lincls -> rotation -> CUB-200 -> Flowers few-shot)
GPU=0 FRAMEWORK=simclr EXPERIMENT=relpred bash pred_ssl/scripts/run_pipeline.sh

# Full matrix via SLURM (edit #SBATCH placeholders first)
ARCH=resnet50 EPOCHS=500 bash pred_ssl/scripts/launch_matrix.sh

# Collect results
python pred_ssl/scripts/extract_results.py --logs-dir ./pred_ssl/logs --out ./pred_ssl/results.csv
```

Pretraining defaults match each framework's original (`main_*.py`): SimCLR/BYOL
`lr 0.3` ×bs/256, cosine; MoCo/LooC `lr 0.03`, step-decay [300,400]; bs 256, 500 epochs.

## Data paths (defaults)
All datasets live under `pred_ssl/datasets/` (run commands from the folder containing `pred_ssl/`):
- ImageNet-100: `./pred_ssl/datasets/imagenet100`   (`train/`, `val/`)
- CUB-200: `./pred_ssl/datasets/cub200_prepared`     (`train/`, `val/`)
- Flowers-102: `./pred_ssl/datasets/flowers102_prepared` (`train/`, `test/`)

Override any of them in relctl's **Runtime** group, or via the `IN100`/`CUB`/`FLOWERS`
env vars for the scripts.

## Pretraining monitor, best/last checkpoints & curves
- **kNN val monitor** (default ON, `knn_eval_freq: 5`): every N pretraining epochs the
  frozen features are scored with a weighted kNN on the val split and logged as
  `KNN_Acc: x%` — the "validation accuracy per epoch" curve SSL otherwise lacks.
- **Checkpoints**: `checkpoint_last.pth.tar` is written every `save_freq` epochs
  (default 10; rolling — what SLURM resumes from), `checkpoint_best.pth.tar` whenever
  the monitored metric improves (kNN acc, or lowest train loss when the monitor is
  off), plus the usual milestones / final `checkpoint_<epochs>.pth.tar` for the evals.
  Writes happen on a background thread (`async_checkpoint: true`) from a decoupled CPU
  snapshot with an atomic temp→rename, so disk I/O never blocks training and a killed
  job never leaves a corrupt checkpoint; set `async_checkpoint: false` for synchronous.
- **Curves (one run)**: `python -m pred_ssl.scripts.plot_curves pred_ssl/logs/<tag>.log
  pred_ssl/logs/<tag>.eval.log` → per-epoch CSV (+ PNG if matplotlib is installed)
  of pretrain losses, `KNN_Acc`, and each linear probe's per-epoch Val loss/acc.
- **Progression (one plot per model, all seeds)**:
  `python -m pred_ssl.scripts.plot_progression --logs-dir pred_ssl/logs --out-dir pred_ssl/curves`
  → one figure per (framework, variant) with the per-epoch mean and a ±1 std band
  over seeds (kNN accuracy + loss); `--metrics`, `--show-seeds`, `--min-seeds` tune it.

## Tests
```bash
python -m pytest pred_ssl/tests/ -q     # CPU-only; no GPU/data needed
```
