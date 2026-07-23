"""
Single source of truth for every knob and action the relctl control panel exposes.

`KNOBS` is metadata only (type / valid range / where it lives / which CLI flag, if
any, reaches it). The LIVE default *values* for train-domain knobs are read from
configs/*.yaml at runtime (see config.py) so they can never drift from the committed
configs; the `default` field here is documentation + the target of `relctl --validate`.

Domains
-------
train         : part of the train.py merged config (base <- framework <- experiment
                <- overlay <- CLI). Reaches train.py via `cli_flag` if set, else via
                the YAML overlay (--config-overlay).
eval_lincls   : CLI args to pred_ssl.eval.linear_probe
eval_fewshot  : CLI args to pred_ssl.eval.few_shot
pilot         : env vars consumed by scripts/run_pilot.sh
gate          : CLI args to scripts/check_pilot_gate.py
runtime       : GPU / conda env (env vars for the scripts)
paths         : dataset locations (env vars for the scripts)
matrix        : grid selectors for scripts/launch_matrix.sh
"""

from dataclasses import dataclass, field
from typing import Optional


FRAMEWORKS = ["simclr", "moco", "byol", "looc", "vicreg"]
EXPERIMENTS = ["baseline", "relpred", "relpred_lambda0", "relpred_decoupled", "relpred_proj3",
               "relpred_proj6", "relpred_split", "relpred_split_80_10_10",
               "relpred_split_45_45_10"]

# Canonical factor order (mirror of data/transforms.py FACTORS). The head emits one
# logit per factor; this set is owned by the data layer and is not a tunable knob.
FACTORS = ["rotation", "hflip", "brightness", "contrast",
           "saturation", "hue", "grayscale", "blur", "crop"]
DELTA_KEYS = ["brightness", "contrast", "saturation", "hue", "blur", "crop"]

# Editor sections, in display order.
GROUPS = [
    ("data",    "Data & run"),
    ("model",   "Model / framework"),
    ("optim",   "Optimizer & LR"),
    ("aug",     "Augmentation"),
    ("rel",     "Relational head"),
    ("eval",    "Eval / probe knobs"),
    ("runtime", "Runtime, paths & pilot"),
]


@dataclass
class Knob:
    key: str
    group: str
    domain: str
    type: str                       # int float bool enum path str list_int list_float list_str dict_float
    default: object
    valid: object = None            # enum -> list of choices; numeric -> (min, max) (None = unbounded)
    cli_flag: Optional[str] = None  # flag that carries it (train.py or an eval/gate script); None+train => overlay
    fw_scope: Optional[str] = None  # framework name if framework-specific
    doc: str = ""
    coupling: str = ""
    note: str = ""                  # e.g. "dead knob, ignored", "len locked at 8"

    @property
    def yaml_only(self):
        """A train-domain knob with no CLI flag -> must travel via the overlay."""
        return self.domain == "train" and self.cli_flag is None


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------

KNOBS = [
    # ---- Data & run (train domain) ----
    Knob("data", "data", "train", "path", "./pred_ssl/datasets/imagenet100", cli_flag="--data",
         doc="Pretraining dataset root (ImageFolder; loader appends /train)."),
    Knob("epochs", "data", "train", "int", 500, valid=(1, None), cli_flag="--epochs",
         doc="Pretraining epochs (also the cosine-LR and BYOL tau denominators)."),
    Knob("batch_size", "data", "train", "int", 256, valid=(1, None), cli_flag="--batch-size",
         coupling="base_lr = lr x batch_size/256 when lr_scale_by_batch=ON",
         doc="Pretraining batch size (DataLoader drop_last=True)."),
    Knob("workers", "data", "train", "int", 8, valid=(0, None), cli_flag="--workers",
         doc="DataLoader workers (persistent_workers only when >0)."),
    Knob("seed", "data", "train", "int", 42, cli_flag="--seed",
         doc="Seeds random/numpy/torch in train.py."),
    Knob("save_dir", "data", "train", "path", "./checkpoints/default", cli_flag="--save-dir",
         doc="Where checkpoint_<epoch>.pth.tar is written."),
    Knob("save_freq", "data", "train", "int", 10, valid=(1, None), cli_flag="--save-freq",
         doc="Checkpoint cadence in epochs: rolling checkpoint_last (always) + milestones "
             "(unless --save-latest); the final epoch and checkpoint_best are always saved."),
    Knob("async_checkpoint", "data", "train", "bool", True,
         doc="Write checkpoints on a background thread (decoupled CPU snapshot + atomic "
             "rename) so disk I/O never blocks training; false = fully synchronous."),
    Knob("print_freq", "data", "train", "int", 20, valid=(1, None), cli_flag="--print-freq",
         doc="In-epoch progress line every N iters (epoch summary always printed)."),

    # ---- Model / framework (train domain) ----
    Knob("arch", "model", "train", "enum", "resnet50", valid=["resnet18", "resnet50"],
         cli_flag="--arch", doc="Backbone (feat_dim 2048/512). Stored in the checkpoint."),
    # framework-specific blocks (all YAML-only; shown only for the active framework)
    Knob("simclr_dim", "model", "train", "int", 128, valid=(1, None), fw_scope="simclr",
         doc="SimCLR projector output dim."),
    Knob("temperature", "model", "train", "float", 0.5, valid=(0.0, None), fw_scope="simclr",
         doc="NT-Xent temperature."),
    Knob("moco_dim", "model", "train", "int", 128, valid=(1, None), fw_scope="moco",
         doc="MoCo projector dim."),
    Knob("moco_k", "model", "train", "int", 16384, valid=(1, None), fw_scope="moco",
         doc="MoCo queue size."),
    Knob("moco_m", "model", "train", "float", 0.999, valid=(0.0, 1.0), fw_scope="moco",
         doc="MoCo key-encoder EMA momentum."),
    Knob("moco_t", "model", "train", "float", 0.2, valid=(0.0, None), fw_scope="moco",
         doc="MoCo softmax temperature."),
    Knob("mlp", "model", "train", "bool", True, fw_scope="moco",
         note="DEAD knob: not read by moco.py (the v2 MLP head is unconditional).",
         doc="Legacy MoCo-v2 MLP flag (ignored)."),
    Knob("proj_hidden_dim", "model", "train", "int", 4096, valid=(1, None), fw_scope="byol",
         doc="BYOL projector hidden dim."),
    Knob("proj_dim", "model", "train", "int", 256, valid=(1, None), fw_scope="byol",
         doc="BYOL projector/predictor output dim."),
    Knob("tau_base", "model", "train", "float", 0.996, valid=(0.0, 1.0), fw_scope="byol",
         doc="BYOL base target-EMA tau (cosine-ramped to 1.0)."),
    Knob("dim", "model", "train", "int", 128, valid=(1, None), fw_scope="looc",
         doc="LooC projector dim."),
    Knob("K", "model", "train", "int", 16384, valid=(1, None), fw_scope="looc",
         doc="LooC queue size."),
    Knob("m", "model", "train", "float", 0.999, valid=(0.0, 1.0), fw_scope="looc",
         doc="LooC key-encoder EMA momentum."),
    Knob("T", "model", "train", "float", 0.2, valid=(0.0, None), fw_scope="looc",
         doc="LooC softmax temperature."),
    Knob("n_aug", "model", "train", "int", 0, valid=(0, 0), fw_scope="looc",
         note="GUARD: any value != 0 raises NotImplementedError (Phase 5).",
         doc="LooC augmentation-specific heads (v1 supports 0 only)."),
    Knob("full_multiview", "model", "train", "bool", False, fw_scope="looc",
         note="GUARD: true raises NotImplementedError (Phase 5).",
         doc="LooC full multiview (v1 supports false only)."),
    Knob("vicreg_expander_layers", "model", "train", "int", 3, valid=(1, None), fw_scope="vicreg",
         doc="Number of Linear layers in the VICReg expander (canonical: 3)."),
    Knob("vicreg_expander_dim", "model", "train", "int", 8192, valid=(1, None), fw_scope="vicreg",
         doc="Hidden width of the VICReg expander (canonical: 8192)."),
    Knob("vicreg_proj_dim", "model", "train", "int", 8192, valid=(1, None), fw_scope="vicreg",
         doc="VICReg expander output dim (canonical: 8192)."),
    Knob("vicreg_sim_coeff", "model", "train", "float", 25.0, valid=(0.0, None), fw_scope="vicreg",
         doc="VICReg invariance (MSE) term weight."),
    Knob("vicreg_std_coeff", "model", "train", "float", 25.0, valid=(0.0, None), fw_scope="vicreg",
         doc="VICReg variance (std hinge) term weight."),
    Knob("vicreg_cov_coeff", "model", "train", "float", 1.0, valid=(0.0, None), fw_scope="vicreg",
         doc="VICReg covariance (decorrelation) term weight."),
    # ---- Projection head (all frameworks) ----
    Knob("proj_preset", "model", "train", "enum", "native", valid=["native", "custom"],
         coupling="native -> each framework's original head (byte-identical); custom -> proj_* MLP",
         doc="Projection-head preset. 'custom' applies proj_layers/hidden/out/bn to all frameworks."),
    Knob("proj_layers", "model", "train", "int", 2, valid=(1, None),
         coupling="only used when proj_preset == custom",
         doc="Number of Linear layers in the custom projection head."),
    Knob("proj_hidden", "model", "train", "int", 2048, valid=(1, None),
         coupling="only used when proj_preset == custom",
         doc="Hidden width of the custom projection head."),
    Knob("proj_out", "model", "train", "int", 256, valid=(1, None),
         coupling="only used when proj_preset == custom; drives MoCo/LooC queue + BYOL predictor dims",
         doc="Output dim of the custom projection head."),
    Knob("proj_bn", "model", "train", "bool", True,
         coupling="only used when proj_preset == custom",
         doc="BatchNorm between layers of the custom projection head."),

    # ---- Optimizer & LR (train domain) ----
    Knob("optimizer", "optim", "train", "enum", "sgd", valid=["sgd", "lars"],
         coupling="vicreg uses lars (its unnormalized loss diverges under plain SGD)",
         doc="Optimizer: sgd, or lars (bias/norm excluded from wd + trust-ratio adaptation)."),
    Knob("lr", "optim", "train", "float", 0.3, valid=(0.0, None), cli_flag="--lr",
         coupling="overridden per framework: simclr/byol 0.3, moco/looc 0.03, vicreg 0.2",
         doc="Base LR before batch scaling."),
    Knob("lr_scale_by_batch", "optim", "train", "bool", True,
         coupling="ON for simclr/byol, OFF for moco/looc",
         doc="If ON: base_lr = lr x batch_size/256 (linear scaling)."),
    Knob("lr_schedule", "optim", "train", "enum", "cosine", valid=["cosine", "step"],
         coupling="`schedule` milestones used only when this is 'step'",
         doc="LR decay shape: cosine over epochs, or step x0.1 at milestones."),
    Knob("schedule", "optim", "train", "list_int", [300, 400],
         coupling="only used when lr_schedule == step",
         doc="Step-decay milestone epochs (x0.1 each)."),
    Knob("warmup_epochs", "optim", "train", "int", 0, valid=(0, None),
         coupling="set to 10 for vicreg; required for lars stability",
         doc="Linear LR warmup over the first N epochs (0 = off)."),
    Knob("momentum", "optim", "train", "float", 0.9, valid=(0.0, 1.0),
         doc="SGD momentum (distinct from MoCo/LooC EMA momentum)."),
    Knob("weight_decay", "optim", "train", "float", 1.0e-4, valid=(0.0, None),
         doc="SGD weight decay."),

    # ---- Augmentation (train domain) ----
    Knob("aug_sharing", "aug", "train", "bool", True,
         coupling="set by experiment; needs rel_lambda>0 to feed the head",
         doc="ON -> RelPairTransform (per-factor same/diff + labels); OFF -> standard two-view."),
    Knob("crop_scale", "aug", "train", "list_float", [0.2, 1.0], valid=(0.0, 1.0),
         doc="RandomResizedCrop scale range (the independent contrastive signal)."),
    Knob("color_strength", "aug", "train", "float", 1.0, valid=(0.0, None), cli_flag="--color-strength",
         doc="Color-jitter strength multiplier (brightness/contrast/saturation/hue ranges)."),
    Knob("p_same", "aug", "train", "float", 0.5, valid=(0.0, 1.0),
         doc="Per-factor same/different coin probability."),
    Knob("blur_mode", "aug", "train", "enum", "sigma", valid=["sigma", "binary"], cli_flag="--blur-mode",
         doc="'different' blur label by sigma gap, or by applied/not-applied flip."),
    Knob("use_rotation", "aug", "train", "bool", False,
         note="Only used by the standard (aug_sharing=OFF) transform.",
         doc="Add RandomRotation90 to the baseline transform."),
    Knob("use_color", "aug", "train", "bool", True,
         note="Only used by the standard (aug_sharing=OFF) transform.",
         doc="Enable ColorJitter+grayscale in the baseline transform."),

    # ---- Relational head (train domain) ----
    Knob("rel_lambda", "rel", "train", "float", 0.5, valid=(0.0, None), cli_flag="--rel-lambda",
         coupling="MASTER switch; set by experiment; >0 builds RelHead (~1.5x cost on moco/looc)",
         doc="Weight of the relational BCE: total = ssl + rel_lambda x rel_loss."),
    Knob("rel_head_hidden", "rel", "train", "int", 2048, valid=(1, None),
         doc="Hidden width of the 3-layer LayerNorm RelHead MLP (used only when rel_lambda>0)."),
    Knob("rel_decoupled", "rel", "train", "bool", False,
         coupling="set by experiment relpred_decoupled; needs rel_lambda>0; +2 backbone forwards/step",
         doc="Decouple the head from the contrastive loss: a STANDARD independent SSL pair plus a "
             "SEPARATE per-factor shared/different pair that feeds only the head."),
    Knob("feat_split", "rel", "train", "bool", False,
         coupling="set by experiment relpred_split; only meaningful with rel_lambda>0",
         doc="Partition h into [vanilla|common|rel] blocks: the SSL head sees vanilla+common, "
             "the relational head sees common+rel (gradient-level disentanglement)."),
    Knob("split_ratios", "rel", "train", "list_float", [0.5, 0.25, 0.25], valid=(0.0, 1.0),
         coupling="used only when feat_split=ON",
         doc="[vanilla, common, rel] fractions of feat_dim; 3 non-negative values summing to 1."),
    Knob("split_decov_lambda", "rel", "train", "float", 0.0, valid=(0.0, None),
         coupling="used only when feat_split=ON and rel_lambda>0",
         doc="Weight of the cross-correlation penalty between the vanilla- and rel-EXCLUSIVE "
             "blocks (0 = off; O(1) is a sensible starting point)."),
    Knob("delta", "rel", "train", "dict_float", {"brightness": 0.2, "contrast": 0.2,
                                                 "saturation": 0.2, "hue": 0.05, "blur": 0.4,
                                                 "crop": 0.4},
         valid=(0.0, None),
         note="All 6 keys required (a missing key KeyErrors at runtime).",
         doc="Per-factor minimum 'different' gap for the continuous factors; for crop it is "
             "the MAX IoU allowed between 'different' boxes (keep >= crop_scale[0])."),

    # ---- Pretraining kNN monitor (train domain, shown with the eval knobs) ----
    Knob("knn_eval_freq", "eval", "train", "int", 5, valid=(0, None),
         coupling="drives checkpoint_best.pth.tar (0 -> best falls back to lowest train loss)",
         doc="Run the kNN val probe every N pretraining epochs (0 = off); logs 'KNN_Acc: x%'."),
    Knob("knn_k", "eval", "train", "int", 20, valid=(1, None),
         doc="Neighbours for the kNN val probe."),
    Knob("knn_temp", "eval", "train", "float", 0.07, valid=(0.0, None),
         doc="Softmax temperature of the weighted kNN vote (InstDisc/MoCo protocol)."),
    Knob("knn_per_class", "eval", "train", "int", 100, valid=(1, None),
         doc="Train-bank images per class for the kNN val probe."),

    # ---- Eval / probe knobs ----
    Knob("eval_epochs", "eval", "eval_lincls", "int", 200, valid=(1, None), cli_flag="--epochs",
         doc="Linear-probe epochs (pipeline EVAL_EPOCHS)."),
    Knob("lincls_lr", "eval", "eval_lincls", "float", 30.0, valid=(0.0, None), cli_flag="--lr",
         doc="Linear-probe LR (high by convention; frozen backbone)."),
    Knob("lincls_batch_size", "eval", "eval_lincls", "int", 256, valid=(1, None), cli_flag="--batch-size",
         doc="Linear-probe batch size."),
    Knob("lincls_momentum", "eval", "eval_lincls", "float", 0.9, valid=(0.0, 1.0), cli_flag="--momentum",
         doc="Linear-probe SGD momentum."),
    Knob("lincls_weight_decay", "eval", "eval_lincls", "float", 0.0, valid=(0.0, None),
         cli_flag="--weight-decay", doc="Linear-probe weight decay."),
    Knob("lincls_schedule", "eval", "eval_lincls", "list_int", [120, 160], cli_flag="--schedule",
         doc="Linear-probe step-decay milestones."),
    Knob("lincls_workers", "eval", "eval_lincls", "int", 8, valid=(0, None), cli_flag="--workers",
         doc="Linear-probe DataLoader workers."),
    Knob("n_shots", "eval", "eval_fewshot", "list_int", [5, 10], cli_flag="--n-shots",
         doc="Few-shot K values (Flowers-102)."),
    Knob("n_trials", "eval", "eval_fewshot", "int", 10, valid=(1, None), cli_flag="--n-trials",
         doc="Few-shot trials per K (for the CI)."),
    Knob("fewshot_lr", "eval", "eval_fewshot", "float", 0.03, valid=(0.0, None), cli_flag="--lr",
         doc="Few-shot linear-classifier LR (Adam)."),
    Knob("fewshot_iterations", "eval", "eval_fewshot", "int", 250, valid=(1, None), cli_flag="--iterations",
         doc="Few-shot classifier iterations."),
    Knob("fewshot_batch_size", "eval", "eval_fewshot", "int", 64, valid=(1, None), cli_flag="--batch-size",
         doc="Few-shot feature-extraction batch size."),
    # gate thresholds
    Knob("gate_leak", "eval", "gate", "float", 98.0, valid=(0.0, 100.0), cli_flag="--leak",
         doc="Per-factor LEAK threshold (% pinned at ~100 -> shortcut)."),
    Knob("gate_stuck", "eval", "gate", "float", 52.0, valid=(0.0, 100.0), cli_flag="--stuck",
         doc="Per-factor STUCK threshold (% near chance)."),
    Knob("gate_min_learning", "eval", "gate", "int", 3, valid=(0, 8), cli_flag="--min-learning",
         doc="Min factors that must be learning to PASS."),
    Knob("gate_min_ssl_drop", "eval", "gate", "float", 0.02, valid=(0.0, 1.0), cli_flag="--min-ssl-drop",
         doc="Min fractional SSL-loss drop to PASS."),

    # ---- Runtime, paths & pilot ----
    Knob("GPU", "runtime", "runtime", "int", 0, valid=(0, None),
         doc="CUDA device index (CUDA_VISIBLE_DEVICES)."),
    Knob("conda_env", "runtime", "runtime", "str", "pytorch_2_0_0",
         doc="Conda env the scripts activate."),
    Knob("IN100", "runtime", "paths", "path", "./pred_ssl/datasets/imagenet100",
         doc="ImageNet-100 root (pipeline/pilot)."),
    Knob("CUB", "runtime", "paths", "path", "./pred_ssl/datasets/cub200_prepared",
         doc="CUB-200 root (pipeline STEP 4)."),
    Knob("FLOWERS", "runtime", "paths", "path", "./pred_ssl/datasets/flowers102_prepared",
         doc="Flowers-102 root (pipeline STEP 5)."),
    Knob("pilot_classes", "runtime", "pilot", "int", 20, valid=(1, None),
         doc="Pilot subset: classes (make_pilot_subset --n-classes)."),
    Knob("pilot_per_class", "runtime", "pilot", "int", 500, valid=(1, None),
         doc="Pilot subset: images per class (--n-per-class)."),
    Knob("pilot_epochs", "runtime", "pilot", "int", 50, valid=(1, None),
         doc="Pilot pretraining epochs."),
]

KNOBS_BY_KEY = {k.key: k for k in KNOBS}

# Framework -> the YAML keys its model actually reads (drives the editor's framework block
# and `relctl --validate`).
FRAMEWORK_KNOBS = {
    "simclr": ["simclr_dim", "temperature"],
    "moco": ["moco_dim", "moco_k", "moco_m", "moco_t", "mlp"],
    "byol": ["proj_hidden_dim", "proj_dim", "tau_base"],
    "looc": ["dim", "K", "m", "T", "n_aug", "full_multiview"],
    "vicreg": ["vicreg_expander_layers", "vicreg_expander_dim", "vicreg_proj_dim",
               "vicreg_sim_coeff", "vicreg_std_coeff", "vicreg_cov_coeff"],
}


def knobs_in_group(group, framework=None):
    """Knobs for an editor section. Framework-specific knobs are filtered to `framework`."""
    out = []
    for k in KNOBS:
        if k.group != group:
            continue
        if k.fw_scope is not None and k.fw_scope != framework:
            continue
        out.append(k)
    return out


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@dataclass
class Action:
    key: str
    label: str
    doc: str
    background: bool = True          # launch under nohup/tmux (long-running) vs run inline
    needs: str = ""                  # human prerequisite note


ACTIONS = [
    Action("pilot", "Pilot + gate",
           "Symlink a small IN-100 subset, short ResNet-18 relpred run, then the automated gate.",
           needs="IN-100 data, GPU"),
    Action("pipeline", "Full pipeline",
           "Pretrain -> IN-100 lincls -> rotation -> CUB-200 -> Flowers few-shot (one log).",
           needs="IN-100 (+CUB/Flowers for steps 4-5), GPU"),
    Action("pretrain", "Pretrain only",
           "STEP 1 only: pretrain the selected (framework, experiment).",
           needs="IN-100 data, GPU"),
    Action("eval", "Eval only (all 4)",
           "STEP 2-5 against an existing checkpoint (run_pipeline.sh MODE=eval).",
           needs="a pretrained checkpoint"),
    Action("eval_in100", "Single eval: IN-100 object",
           "Linear probe on ImageNet-100 (object classification).",
           needs="checkpoint + IN-100"),
    Action("eval_rotation", "Single eval: IN-100 rotation",
           "4-way rotation linear probe on ImageNet-100.",
           needs="checkpoint + IN-100"),
    Action("eval_cub", "Single eval: CUB-200",
           "Linear probe on CUB-200 (object classification).",
           needs="checkpoint + CUB-200"),
    Action("eval_flowers", "Single eval: Flowers few-shot",
           "Few-shot eval on Flowers-102.",
           needs="checkpoint + Flowers-102"),
    Action("resume", "Resume pretraining",
           "Continue a pretrain from a checkpoint_*.pth.tar.",
           needs="an existing checkpoint"),
    Action("matrix", "Matrix (SLURM grid)",
           "Submit frameworks x experiments via sbatch (pretrain + eval with afterok dep).",
           background=False, needs="SLURM (sbatch), filled #SBATCH placeholders"),
    Action("make_subset", "Make pilot subset",
           "Symlink an IN-100 subset (no training).", background=False, needs="IN-100 data"),
    Action("gate_check", "Run pilot gate on a log",
           "Evaluate the Phase-2 gate against a pilot log.", background=False, needs="a pilot log"),
    Action("extract", "Extract results -> CSV",
           "Parse pred_ssl/logs/*.log into results.csv.", background=False),
    Action("tests", "Run unit tests",
           "CPU-only pytest suite (no GPU/data).", background=False),
]

ACTIONS_BY_KEY = {a.key: a for a in ACTIONS}
