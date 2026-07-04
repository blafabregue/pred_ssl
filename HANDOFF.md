# pred_ssl — Handoff & Reproduction Guide

A complete, self-contained guide to pull the code, place the datasets, and run
**everything** in `pred_ssl` (the relational / pairwise augmentation-prediction SSL
project). Written for running on a Linux GPU/SLURM server.

> TL;DR — clone the repo, drop three dataset folders into the repo root, create the
> conda env, then drive everything from one interactive panel:
> `python -m pred_ssl.relctl`.

---

## 1. What this is

`pred_ssl` is a unified, config-driven self-supervised-learning (SSL) codebase. It
pretrains a ResNet backbone with one of five SSL frameworks (**SimCLR / MoCo / BYOL /
LooC / VICReg**) plus an optional **relational head** that predicts, per augmentation factor,
whether the same parameter was applied to both views. It then evaluates the frozen
backbone with linear probes (ImageNet-100 object + rotation, CUB-200) and few-shot
(Flowers-102).

Everything is driven through one Python entrypoint (`train.py`) + eval modules, wrapped
by helper scripts, and surfaced through an interactive control panel, **relctl**.
`pred_ssl` is standalone — it imports only itself; it just *reuses the dataset folders*.

---

## 2. Get the code

The repo IS the `pred_ssl` Python package — cloning it gives you a `pred_ssl/` folder
that is the package itself (it imports only itself).

```bash
git clone https://github.com/blafabregue/pred_ssl.git
```

The datasets live **inside** `pred_ssl/` (at `pred_ssl/datasets/`). Run every command
from the folder that *contains* `pred_ssl/` — i.e. the parent of the clone, so the
`pred_ssl` package resolves — not from inside `pred_ssl/`:

```
<working dir>/           # <-- run all commands from here (parent of the clone)
└── pred_ssl/            # the cloned repo == the Python package
    ├── train.py, relctl/, eval/, scripts/, configs/ ...   # code, at the root
    └── datasets/        # the three datasets go here (Section 3)
```

The actual data + checkpoints are **git-ignored** (too large), so they are *not* in the
clone — only the `pred_ssl/datasets/README.md` placeholder is. You add the data (Section 3).

---

## 3. Datasets

All three datasets live under **`pred_ssl/datasets/`**. Each is a standard `ImageFolder`
tree (`split/<class>/*.jpg`).

| Dataset | Used for | Location (relative to the working dir) | Sub-folders |
|---|---|---|---|
| **ImageNet-100** | pretraining + IN-100 linear/rotation eval | `./pred_ssl/datasets/imagenet100/` | `train/`, `val/` |
| **CUB-200-2011** | CUB-200 linear eval | `./pred_ssl/datasets/cub200_prepared/` | `train/`, `val/` |
| **Flowers-102** | few-shot eval | `./pred_ssl/datasets/flowers102_prepared/` | `train/`, `test/` |

Final layout the code expects:

```
<working dir>/                           # run all commands from here
└── pred_ssl/
    ├── ...                              # code
    └── datasets/
        ├── imagenet100/                 # ImageNet-100
        │   ├── train/<synset>/*.JPEG
        │   └── val/<synset>/*.JPEG
        ├── cub200_prepared/             # CUB-200
        │   ├── train/<class>/*.jpg
        │   └── val/<class>/*.jpg
        └── flowers102_prepared/         # Flowers-102
            ├── train/<class>/*.jpg
            └── test/<class>/*.jpg
```

### 3a. Packaging + uploading to Google Drive (the person who HAS the data)

The datasets are JPEG images (already compressed), so use a **plain, uncompressed
`tar`** — much faster to create, ~same size. From the repo root:

```bash
tar cf cub200_prepared.tar     -C moco cub200_prepared
tar cf flowers102_prepared.tar -C .    flowers102_prepared

# imagenet100 is a symlink here, and its target contains a self-referential nested
# copy — tar the REAL directory and exclude that nested loop:
tar cf imagenet100.tar -C /home/<you>/projects --exclude='imagenet100/imagenet100' imagenet100
```

Upload all three `.tar` files into **one** Google Drive folder (e.g. `pred_ssl_datasets`),
then share the folder as **"Anyone with the link"** and put that folder link below:

> Drive folder: https://drive.google.com/drive/folders/1eK5vRp2vKaW7-ug396wifMviUIM0VclR

### 3b. Downloading + placing the data (the professor)

```bash
cd ravan_internship           # the working dir (contains pred_ssl/)
pip install gdown             # one-time

# grab all three archives from the shared Drive folder
gdown --folder "https://drive.google.com/drive/folders/1eK5vRp2vKaW7-ug396wifMviUIM0VclR"

# extract all three INTO ./pred_ssl/datasets/  (tar xf auto-detects format)
tar xf imagenet100.tar         -C pred_ssl/datasets   # -> pred_ssl/datasets/imagenet100/
tar xf cub200_prepared.tar     -C pred_ssl/datasets   # -> pred_ssl/datasets/cub200_prepared/
tar xf flowers102_prepared.tar -C pred_ssl/datasets   # -> pred_ssl/datasets/flowers102_prepared/
```

Verify the layout (must print three "OK" lines):

```bash
for p in pred_ssl/datasets/imagenet100/train pred_ssl/datasets/cub200_prepared/train pred_ssl/datasets/flowers102_prepared/train; do
  [ -d "$p" ] && echo "OK   $p" || echo "MISSING $p"
done
```

### 3c. If the data lives elsewhere (e.g. a scratch/data disk)

You do **not** have to place data in the repo root. Point the tools at any location:

- In **relctl**: open group **7) Runtime, paths & pilot** and set `IN100`, `CUB`,
  `FLOWERS` to your paths.
- For the **scripts**: pass them as env vars, e.g.
  `IN100=/data/imagenet100 CUB=/data/cub200_prepared FLOWERS=/data/flowers102_prepared ...`

### 3d. (Optional) Regenerate "prepared" datasets from raw

If you only have the raw datasets, the prep scripts that produced the above are:
`Moco-Imagenet/extract_imagenet100.py`, `moco/prepare_cub.py`,
`flowers102_raw/prepare_flowers102.py`. (Not needed if you use the Drive archives.)

---

## 4. Environment setup

The scripts default to a conda env named `pytorch_2_0_0`. Use it if it exists,
otherwise create an equivalent (any name works — override with `CONDA_ENV=<name>`):

```bash
conda create -n pytorch_2_0_0 python=3.11 -y
conda activate pytorch_2_0_0
pip install -r pred_ssl/requirements.txt                 # core (required)
pip install -r pred_ssl/requirements-dev.txt             # optional: nicer UI (rich) + tests
```

(Equivalently, the explicit packages: `torch torchvision numpy pillow pyyaml` for the
core, plus `rich pytest` for the extras. If `pip install torch` hits a CUDA mismatch on
the cluster, install torch/torchvision from the official PyTorch channel for your CUDA
build instead — https://pytorch.org/get-started/locally/ — then `pip install -r` the rest.)

Quick check:

```bash
python -c "import torch, torchvision, yaml; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

(The first `import torch` can take several seconds — that's normal.)

---

## 5. Sanity checks (no GPU/data needed)

```bash
# (a) the control panel's knob catalog matches the configs + argparse
python -m pred_ssl.relctl --validate
#   -> "OK — catalog is in sync ..."

# (b) unit tests (CPU-only; needs pytest)
python -m pytest pred_ssl/tests/ -q

# (c) preview what a run WOULD do, without running it (per mode)
MODE=pilot    bash pred_ssl/run.sh --dry-run
MODE=pipeline bash pred_ssl/run.sh --dry-run
```

`--dry-run` prints a prerequisite checklist (env, GPU, datasets) — a fast way to
confirm the datasets are found before launching anything.

---

## 6. Running everything — the interactive panel (recommended)

```bash
python -m pred_ssl.relctl            # auto: Rich UI if installed, else plain
python -m pred_ssl.relctl --plain    # force the plain (zero-dependency) UI
```

It's a **typed-key menu**: type the letter/number shown, press **Enter**. The top
banner always shows the current plan (`action`, `fw`, `exp`, `arch`, `epochs`,
`base_lr`, how many edits, running jobs).

**Common actions**

| Goal | Keys |
|---|---|
| Change framework (simclr/moco/byol/looc/vicreg) | `f` → number |
| Change experiment (baseline/relpred/relpred_lambda0/relpred_decoupled) | `e` → number |
| Edit a setting | group number `1`–`7` → row number → new value → `b` |
| Set dataset paths | `7` → edit `IN100` / `CUB` / `FLOWERS` |
| Pick what to run | `a` → number |
| Preview & launch | `r` → `n` (background) / `t` (tmux) / `f` (foreground) / `d` (dry-run) |
| Watch running jobs | `j` → `t <#>` tail, `s <#>` stop |
| See results table | `o` → `x` (extract logs → results.csv) |
| Save / load a config profile | `s` / `l` ; reset edits `z` ; verify resolved config `v` |
| Quit (jobs keep running) | `q` |

In an editor: `*` marks an edited knob, `d` diffs vs defaults, `h <#>` shows help.
Editing group **5** then the `delta` row opens a sub-form for its 5 per-factor keys.

**Example — a quick end-to-end smoke run from the panel:**
1. `a` → `Pilot + gate`  (subset pretrain + automated gate)
2. `7` → set `pilot_epochs`=`2`, `pilot_classes`=`5`, `pilot_per_class`=`50` → `b`
3. `r` → `n`  (launches in the background, returns immediately)
4. `j` → watch progress; `t 1` to tail the log

A full "MoCo baseline, 100 epochs, launch pilot" is just:
`f`→moco, `e`→baseline, `1`→edit `epochs`→`100`→`b`, `a`→pilot, `r`→`n`.

---

## 7. Running everything — non-interactive (scripts & SLURM)

For automation/SLURM you can bypass the UI. All commands run from the repo root.

```bash
# Pilot: short ResNet-18 run on an IN-100 subset + automated gate
GPU=0 bash pred_ssl/scripts/run_pilot.sh

# One full pipeline: pretrain -> IN-100 lincls -> rotation -> CUB-200 -> Flowers few-shot
GPU=0 FRAMEWORK=simclr EXPERIMENT=relpred bash pred_ssl/scripts/run_pipeline.sh

# Pretrain only / eval only
MODE=pretrain GPU=0 FRAMEWORK=simclr EXPERIMENT=relpred bash pred_ssl/scripts/run_pipeline.sh
MODE=eval     GPU=0 FRAMEWORK=simclr EXPERIMENT=relpred bash pred_ssl/scripts/run_pipeline.sh

# Direct entrypoints (full control over every flag)
python -m pred_ssl.train --framework simclr --experiment relpred \
    --arch resnet50 --data ./pred_ssl/datasets/imagenet100 --epochs 500 \
    --save-dir ./pred_ssl/checkpoints/simclr_relpred
python -m pred_ssl.eval.linear_probe --data ./pred_ssl/datasets/imagenet100 --arch resnet50 \
    --pretrained ./pred_ssl/checkpoints/simclr_relpred/checkpoint_0500.pth.tar
python -m pred_ssl.eval.linear_probe --data ./pred_ssl/datasets/imagenet100 --eval-rotation \
    --pretrained <ckpt>
python -m pred_ssl.eval.few_shot --data ./pred_ssl/datasets/flowers102_prepared --pretrained <ckpt>

# Inspect the fully-resolved config a run will use (no training)
python -m pred_ssl.train --framework moco --experiment baseline --print-config

# Full SLURM matrix (4 frameworks x experiments). FILL the #SBATCH placeholders first:
#   edit pred_ssl/scripts/sbatch_pretrain.slurm + sbatch_eval.slurm  (<PARTITION>, <ACCOUNT>)
ARCH=resnet50 EPOCHS=500 bash pred_ssl/scripts/launch_matrix.sh

# Collect all results into a CSV
python pred_ssl/scripts/extract_results.py --logs-dir ./pred_ssl/logs --out ./pred_ssl/results.csv
```

Useful env-var overrides for any script: `GPU`, `CONDA_ENV`, `ARCH`, `EPOCHS`,
`EVAL_EPOCHS`, `IN100`, `CUB`, `FLOWERS`, `SAVE_DIR`, `CKPT`.

### The three experiments
- `baseline` — standard independent augmentation, no relational head (reproduces the
  existing per-framework numbers).
- `relpred` — the method: per-factor sharing loader + relational head (`rel_lambda=0.5`).
- `relpred_lambda0` — ablation: sharing loader, head off.
- `relpred_decoupled` — the method without the augmentation confound: standard independent
  contrastive pair + a separate shared/different pair feeding only the head
  (`rel_decoupled=true`; +2 backbone forwards/step).

---

## 8. Outputs

| What | Where |
|---|---|
| Checkpoints | `pred_ssl/checkpoints/<framework>_<experiment>/checkpoint_<epoch>.pth.tar` |
| Run logs | `pred_ssl/logs/<framework>_<experiment>.log` |
| Collected results | `pred_ssl/results.csv` |
| relctl runtime state | `pred_ssl/.relctl/` (jobs registry, generated config overlays) |
| Saved relctl profiles | `pred_ssl/relctl/profiles/` |

---

## 9. Notes & troubleshooting

- **Run from the repo root.** `python -m pred_ssl.relctl` (and the `bash pred_ssl/...`
  scripts) must be launched from the folder that contains `pred_ssl/`, so Python can find
  the `pred_ssl` package. Running from *inside* `pred_ssl/` gives
  `No module named 'pred_ssl'`.
- **Datasets not found.** Check `--dry-run` output; either place the folders at the
  default paths (Section 3) or override `IN100`/`CUB`/`FLOWERS`.
- **SLURM matrix.** Edit the `#SBATCH` `<PARTITION>` / `<ACCOUNT>` placeholders in
  `pred_ssl/scripts/sbatch_pretrain.slurm` and `sbatch_eval.slurm` before
  `launch_matrix.sh`.
- **GPU select.** `GPU=<index>` (sets `CUDA_VISIBLE_DEVICES`).
- **Long runs.** Pretraining is hundreds of epochs; launch in the background from
  relctl (`r` → `n`/`t`) or via SLURM. relctl's job registry survives an SSH drop.
- **Pretraining defaults** match each framework's original: SimCLR/BYOL `lr 0.3`
  (×bs/256, cosine); MoCo/LooC `lr 0.03`, step-decay `[300,400]`; batch 256, 500 epochs,
  ResNet-50.

---

*For panel internals and the full knob/action catalog, see `pred_ssl/README.md`.*
