"""
Single source of truth for the SLURM experiment matrix.

Each experiment is one (framework x variant x seed) pretraining run, evaluated after it
finishes. `slurm_status.py` (the report) and `slurm_submit.sh` (the launcher) both read
the matrix from here, so they can never disagree.

The matrix is env-overridable (so `bash slurm_submit.sh` runs the right subset out of the
box, and you can narrow it without editing code):

    FRAMEWORKS   space-separated (default: simclr moco byol looc vicreg)
    VARIANTS     space-separated (default: all six variants below)
    SEEDS        space-separated (default: 1 2 3 4 5)       # repeats for statistical noise
    ARCH         resnet18 | resnet50 (default: resnet50)
    EPOCHS       pretraining epochs (default: 500)

The default variants (per the study design):
    baseline       vanilla SSL, no relational head            (experiment: baseline)
    relpred        vanilla + the new relational loss          (experiment: relpred)
    relpred_proj3  relpred + the new 3-layer projection head  (experiment: relpred_proj3)
    relpred_proj6  relpred + a 6-layer projection head        (experiment: relpred_proj6)
Opt-in (latent-split / disentanglement study, add via VARIANTS="... relpred_split"):
    relpred_split, relpred_split_80_10_10, relpred_split_45_45_10

Usage:
    python -m pred_ssl.scripts.experiments               # human table
    python -m pred_ssl.scripts.experiments --format tsv  # machine-readable (for bash)
"""

import os

# variant name -> (experiment config in configs/experiment/, one-line description)
VARIANTS = {
    "baseline":      ("baseline",      "vanilla SSL, no relational head"),
    "relpred":       ("relpred",       "vanilla + relational loss"),
    "relpred_proj3": ("relpred_proj3", "relpred + custom 3-layer projection head"),
    "relpred_proj6": ("relpred_proj6", "relpred + custom 6-layer projection head"),
    # Latent-split (disentanglement) variants: kept runnable but OUT of the default
    # matrix — across frameworks they matched or slightly trailed plain relpred, and the
    # three ratio settings were indistinguishable. Run them explicitly with
    # VARIANTS="relpred_split ..." if you want the partition study.
    "relpred_split": ("relpred_split", "relpred + latent split 0.50/0.25/0.25"),
    "relpred_split_80_10_10": ("relpred_split_80_10_10",
                               "relpred + latent split 0.80/0.10/0.10 (vanilla-heavy)"),
    "relpred_split_45_45_10": ("relpred_split_45_45_10",
                               "relpred + latent split 0.45/0.45/0.10 (common-heavy)"),
}

# Default matrix. Narrow/extend with VARIANTS="baseline relpred" (etc.).
DEFAULT_VARIANTS = ["baseline", "relpred", "relpred_proj3", "relpred_proj6"]

DEFAULT_FRAMEWORKS = ["simclr", "moco", "byol", "looc", "vicreg"]
DEFAULT_SEEDS = ["1", "2", "3", "4", "5"]


def _env_list(name, default):
    v = os.environ.get(name, "").strip()
    return v.split() if v else list(default)


def matrix():
    """Return the ordered list of experiment dicts from the (env-overridable) matrix."""
    frameworks = _env_list("FRAMEWORKS", DEFAULT_FRAMEWORKS)
    variants = _env_list("VARIANTS", DEFAULT_VARIANTS)
    seeds = _env_list("SEEDS", DEFAULT_SEEDS)
    arch = os.environ.get("ARCH", "resnet50")
    epochs = int(os.environ.get("EPOCHS", "500"))

    exps = []
    for fw in frameworks:
        for var in variants:
            if var not in VARIANTS:
                raise SystemExit(f"unknown variant '{var}' (known: {', '.join(VARIANTS)})")
            experiment = VARIANTS[var][0]
            for seed in seeds:
                tag = f"{fw}_{var}_{arch}_s{seed}"
                exps.append({
                    "tag": tag,
                    "framework": fw,
                    "variant": var,
                    "experiment": experiment,
                    "arch": arch,
                    "seed": int(seed),
                    "epochs": epochs,
                    "save_dir": f"./pred_ssl/checkpoints/{tag}",
                    "log": f"./pred_ssl/logs/{tag}.log",
                })
    return exps


# TSV column order shared with slurm_submit.sh (keep in sync).
TSV_FIELDS = ["tag", "framework", "experiment", "arch", "seed", "epochs", "save_dir", "log"]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="pred_ssl SLURM experiment matrix")
    ap.add_argument("--format", choices=["human", "tsv"], default="human")
    args = ap.parse_args()

    m = matrix()
    if args.format == "tsv":
        for e in m:
            print("\t".join(str(e[k]) for k in TSV_FIELDS))
        return

    print(f"{len(m)} experiments "
          f"({len({e['framework'] for e in m})} frameworks x "
          f"{len({e['variant'] for e in m})} variants x "
          f"{len({e['seed'] for e in m})} seeds, "
          f"{m[0]['arch']}, {m[0]['epochs']} epochs)\n")
    for e in m:
        print(f"  {e['tag']:<40}  {VARIANTS[e['variant']][1]}")


if __name__ == "__main__":
    main()
