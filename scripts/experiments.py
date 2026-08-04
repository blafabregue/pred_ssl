"""
Single source of truth for the SLURM experiment matrix.

Each experiment is one (framework x variant x seed) pretraining run, evaluated after it
finishes. `slurm_status.py` (the report) and `slurm_submit.sh` (the launcher) both read
the matrix from here, so they can never disagree.

The matrix is env-overridable (so `bash slurm_submit.sh` runs the right subset out of the
box, and you can narrow it without editing code):

    FRAMEWORKS   space-separated (default: simclr moco byol looc vicreg posonly)
    VARIANTS     space-separated (default: the six below + the positives-only study)
    SEEDS        space-separated (default: 1 2)             # repeats for statistical noise
    ARCH         resnet18 | resnet50 (default: resnet50)
    EPOCHS       pretraining epochs (default: 500)

The default variants (everything the paper reports):
    baseline         vanilla SSL, no auxiliary head
    augself          AugSelf (Lee et al. 2021) -- the closest prior work
    relpred_lambda0  sharing loader, head off -- view-distribution control
    relpred          vanilla + the relational loss
    relpred_proj3    relpred + a 3-layer projection head (recommended)
    relpred_proj6    relpred + a 6-layer projection head
    posonly_geom / posonly_color / posonly_all
                     the positives-only study; generated ONLY for framework `posonly`,
                     which in turn only takes these plus baseline (its lambda=0
                     collapse control) and augself. The matrix is a cross-product, so
                     these pairings are enforced by VARIANT_FRAMEWORKS /
                     FRAMEWORK_VARIANTS rather than left to the caller to remember.
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
    "augself":       ("augself",       "AugSelf baseline (Lee et al. 2021), closest prior work"),
    "essl":          ("essl",          "E-SSL baseline (Dangovski et al. 2022), with its safeguards"),
    "extended_essl": ("extended_essl", "our extension of E-SSL's per-view target to all factors"),
    "relpred":       ("relpred",       "vanilla + relational loss"),
    "relpred_proj3": ("relpred_proj3", "relpred + custom 3-layer projection head"),
    "relpred_proj6": ("relpred_proj6", "relpred + custom 6-layer projection head"),
    # THE control for the paper's mechanism claim: same shared/different augmentation
    # distribution as relpred (rotation included), relational head OFF. Any gain it
    # already shows is due to the view distribution, not to the auxiliary loss.
    "relpred_lambda0": ("relpred_lambda0", "sharing loader, head off (view-distribution control)"),
    "relpred_regress": ("relpred_regress", "same views/factors, l2 on parameter differences (target ablation)"),
    # Latent-split (disentanglement) variants: kept runnable but OUT of the default
    # matrix — across frameworks they matched or slightly trailed plain relpred, and the
    # three ratio settings were indistinguishable. Run them explicitly with
    # VARIANTS="relpred_split ..." if you want the partition study.
    "relpred_split": ("relpred_split", "relpred + latent split 0.50/0.25/0.25"),
    "relpred_split_80_10_10": ("relpred_split_80_10_10",
                               "relpred + latent split 0.80/0.10/0.10 (vanilla-heavy)"),
    "relpred_split_45_45_10": ("relpred_split_45_45_10",
                               "relpred + latent split 0.45/0.45/0.10 (common-heavy)"),
    # Positives-only study, where transformation prediction is the ONLY thing
    # opposing collapse. In the default matrix, but generated for the `posonly`
    # framework alone (see VARIANT_FRAMEWORKS below). Read them on effective rank,
    # not on the loss -- both failure modes make the loss go down. The baseline
    # variant doubles as the lambda=0 control and is expected to collapse.
    "posonly_geom":  ("posonly_geom",  "positives-only + geometric factors (crop/rotation/hflip)"),
    "posonly_color": ("posonly_color", "positives-only + photometric factors (falsification arm)"),
    "posonly_all":   ("posonly_all",   "positives-only + all nine factors"),
    "posonly_geom_decov": ("posonly_geom_decov",
                           "posonly_geom + decorrelation safety net (lambda=10)"),
}

# Default matrix: everything the paper reports, so `slurm_status` shows the true
# picture without needing VARIANTS=... to be remembered. Narrow it with
# VARIANTS="baseline relpred" when submitting a subset.
DEFAULT_VARIANTS = [
    "baseline",          # unmodified framework
    "augself",           # closest prior work (Lee et al., 2021)
    "relpred_lambda0",   # view-distribution control for the mechanism claim
    "relpred",           # the method
    "relpred_proj3",     # + 3-layer projector (the recommended operating point)
    "relpred_proj6",     # + 6-layer projector (depth is non-monotone)
    "posonly_geom",      # positives-only study (generated for `posonly` only)
    "posonly_color",     # its falsification arm
    "posonly_all",
    "posonly_geom_decov",  # its fallback arm, if the clean runs collapse
]

DEFAULT_FRAMEWORKS = ["simclr", "moco", "byol", "looc", "vicreg", "posonly"]
DEFAULT_SEEDS = ["1", "2"]

# ---------------------------------------------------------------------------
# Pairing: the matrix is a cross-product, but not every cell means anything.
# ---------------------------------------------------------------------------
# Without these, putting the positives-only study in the defaults would also
# generate simclr_posonly_geom and friends -- which are just relpred_regress with a
# masked factor set, under a name that claims otherwise. Skipping them is what lets
# `posonly` sit in DEFAULT_FRAMEWORKS instead of being an opt-in nobody remembers.

# Variants that only apply to certain frameworks.
VARIANT_FRAMEWORKS = {
    "posonly_geom": {"posonly"},
    "posonly_color": {"posonly"},
    "posonly_all": {"posonly"},
    "posonly_geom_decov": {"posonly"},
}

# Frameworks that only take certain variants. `posonly` exists to ask whether
# transformation prediction alone holds a representation up, so it is paired with
# the variants bearing on that: `baseline` is the lambda=0 collapse control (it is
# SUPPOSED to collapse -- without it a non-collapse proves nothing), and `augself`
# runs the same question with AugSelf's own crop+colour target.
FRAMEWORK_VARIANTS = {
    "posonly": {"baseline", "augself", "posonly_geom", "posonly_color", "posonly_all",
                "posonly_geom_decov"},
}


def applies(framework, variant):
    """Whether this (framework, variant) cell is a meaningful experiment."""
    allowed_frameworks = VARIANT_FRAMEWORKS.get(variant)
    if allowed_frameworks is not None and framework not in allowed_frameworks:
        return False
    allowed_variants = FRAMEWORK_VARIANTS.get(framework)
    if allowed_variants is not None and variant not in allowed_variants:
        return False
    return True


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
            if not applies(fw, var):
                continue
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
    if not exps:
        # An empty matrix from a non-empty request means every cell was skipped by
        # the pairing rules; say so rather than silently submitting nothing.
        raise SystemExit(
            f"no valid (framework, variant) cell in FRAMEWORKS={' '.join(frameworks)} x "
            f"VARIANTS={' '.join(variants)}. The posonly_* variants only apply to "
            f"framework 'posonly', which in turn only takes "
            f"{' '.join(sorted(FRAMEWORK_VARIANTS['posonly']))}.")
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
