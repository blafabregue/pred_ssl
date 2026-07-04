"""
Plan builder: turn (ConfigModel, action) into the exact command(s) relctl will run.

Design choices that keep us faithful to the existing scripts:
  * pilot/pipeline/pretrain/eval go through scripts/run_pilot.sh & run_pipeline.sh so
    the STEP markers + log format extract_results.py parses are preserved, and the
    CONFIG_OVERLAY passthrough we added carries every edited knob.
  * single-eval steps and resume call train.py / the eval modules directly (the
    wrappers can't express "one eval step" or "--resume").
  * arch/epochs/data/save_dir are explicit flags; all other edits ride the overlay.
"""

import os
import shlex
from dataclasses import dataclass, field

from .knobs import KNOBS_BY_KEY, ACTIONS_BY_KEY


def q(v):
    return shlex.quote(str(v))


@dataclass
class Plan:
    action: str
    commands: list                       # shell strings, run in order under one wrapper
    background: bool
    log: str = ""                        # primary log (relative to repo root)
    append_log: bool = False             # append vs truncate the log
    tag: str = ""
    save_dir: str = ""
    ckpt: str = ""                       # checkpoint a run reads (eval/resume)
    overlay_path: str = ""               # "" if no overlay
    overlay_dict: dict = field(default_factory=dict)
    preflight_mode: str = ""             # run.sh mode to scrape for preflight, or ""
    notes: list = field(default_factory=list)


def _ckpt_default(save_dir, epochs):
    return "%s/checkpoint_%04d.pth.tar" % (save_dir, int(epochs))


def _eval_flags(model, keys):
    """CLI flags for eval knobs that the user changed (skip defaults)."""
    out = []
    for k in keys:
        if not model.is_dirty(k):
            continue
        kn = KNOBS_BY_KEY[k]
        val = model.value(k)
        if kn.type in ("list_int", "list_float"):
            out.append("%s %s" % (kn.cli_flag, " ".join(str(x) for x in val)))
        else:
            out.append("%s %s" % (kn.cli_flag, q(val)))
    return out


def build_plan(model, overlay_path=None):
    a = model.action
    fw, exp = model.framework, model.experiment
    arch = model.value("arch")
    epochs = model.value("epochs")
    eval_epochs = model.value("eval_epochs")
    gpu = model.value("GPU")
    conda = model.value("conda_env")
    in100 = model.value("IN100")
    cub = model.value("CUB")
    flowers = model.value("FLOWERS")
    tag = "%s_%s" % (fw, exp)
    save_dir = model.value("save_dir") if model.is_dirty("save_dir") else "./pred_ssl/checkpoints/%s" % tag

    overlay = model.overlay_dict()
    use_overlay = bool(overlay)
    ov_path = overlay_path or "pred_ssl/.relctl/overlays/<pending>.yaml"
    ov_env = ("CONFIG_OVERLAY=%s " % q(ov_path)) if use_overlay else ""

    def base_plan(**kw):
        kw.setdefault("tag", tag)
        kw.setdefault("save_dir", save_dir)
        kw.setdefault("overlay_path", ov_path if use_overlay else "")
        kw.setdefault("overlay_dict", overlay)
        return Plan(action=a, **kw)

    # ---- pilot -------------------------------------------------------------
    if a == "pilot":
        env = ("GPU=%s CONDA_ENV=%s ARCH=%s EPOCHS=%s SRC=%s N_CLASSES=%s N_PER_CLASS=%s "
               "FRAMEWORK=%s %s" % (gpu, q(conda), arch, model.value("pilot_epochs"), q(in100),
                                    model.value("pilot_classes"), model.value("pilot_per_class"),
                                    fw, ov_env))
        return base_plan(commands=[env + "bash pred_ssl/scripts/run_pilot.sh"],
                         background=True, log="./pred_ssl/logs/pilot_%s.log" % fw,
                         preflight_mode="pilot",
                         notes=["pilot forces experiment=relpred, batch_size=256, workers=8 "
                                "(quick ResNet-18 sanity run); EPOCHS uses the pilot_epochs knob",
                                "ends with the automated gate check"])

    # ---- pipeline / pretrain-only / eval-only ------------------------------
    if a in ("pipeline", "pretrain", "eval"):
        mode = {"pipeline": "all", "pretrain": "pretrain", "eval": "eval"}[a]
        env = ("GPU=%s CONDA_ENV=%s FRAMEWORK=%s EXPERIMENT=%s ARCH=%s EPOCHS=%s EVAL_EPOCHS=%s "
               "IN100=%s CUB=%s FLOWERS=%s SAVE_DIR=%s MODE=%s %s" %
               (gpu, q(conda), fw, exp, arch, epochs, eval_epochs, q(in100), q(cub), q(flowers),
                q(save_dir), mode, ov_env))
        ckpt = ""
        notes = []
        if a == "eval":
            ckpt = model.eval_ckpt or _ckpt_default(save_dir, epochs)
            env = "CKPT=%s " % q(ckpt) + env
            notes.append("eval-only re-runs all 4 evals; only EVAL_EPOCHS is tunable here "
                         "(use single-eval actions to change lincls lr/schedule/few-shot knobs)")
        if a == "pretrain":
            notes.append("STEP 1 only; run the 'eval only' action afterwards for STEP 2-5")
        return base_plan(commands=[env + "bash pred_ssl/scripts/run_pipeline.sh"],
                         background=True, log="./pred_ssl/logs/%s.log" % tag,
                         ckpt=ckpt, preflight_mode="pipeline", notes=notes)

    # ---- single eval steps -------------------------------------------------
    if a in ("eval_in100", "eval_rotation", "eval_cub", "eval_flowers"):
        ckpt = model.eval_ckpt or _ckpt_default(save_dir, epochs)
        if a == "eval_flowers":
            flags = _eval_flags(model, ["n_shots", "n_trials", "fewshot_lr",
                                        "fewshot_iterations", "fewshot_batch_size"])
            cmd = ("python -m pred_ssl.eval.few_shot --data %s --pretrained %s --arch %s %s"
                   % (q(flowers), q(ckpt), arch, " ".join(flags))).rstrip()
            log = "./pred_ssl/logs/%s_flowers.log" % tag
        else:
            data = {"eval_in100": in100, "eval_rotation": in100, "eval_cub": cub}[a]
            rot = " --eval-rotation" if a == "eval_rotation" else ""
            flags = _eval_flags(model, ["eval_epochs", "lincls_lr", "lincls_batch_size",
                                        "lincls_momentum", "lincls_weight_decay",
                                        "lincls_schedule", "lincls_workers"])
            cmd = ("python -m pred_ssl.eval.linear_probe --data %s --pretrained %s --arch %s%s %s"
                   % (q(data), q(ckpt), arch, rot, " ".join(flags))).rstrip()
            log = "./pred_ssl/logs/%s_%s.log" % (tag, a.replace("eval_", ""))
        return base_plan(commands=[cmd], background=True, log=log, ckpt=ckpt,
                         overlay_path="", overlay_dict={},
                         notes=["single-eval logs are not part of the canonical "
                                "<fw>_<exp>.log parsed into results.csv"])

    # ---- resume ------------------------------------------------------------
    if a == "resume":
        ckpt = model.resume_ckpt
        ov_flag = (" --config-overlay %s" % q(ov_path)) if use_overlay else ""
        train = ("python -m pred_ssl.train --framework %s --experiment %s --arch %s --data %s "
                 "--epochs %s --save-dir %s --resume %s%s" %
                 (fw, exp, arch, q(model.value("data")), epochs, q(save_dir), q(ckpt), ov_flag))
        return base_plan(commands=['echo "STEP 1: Pretrain (resume from %s)"' % ckpt, train],
                         background=True, log="./pred_ssl/logs/%s.log" % tag, append_log=True,
                         ckpt=ckpt,
                         notes=["resume appends to the existing log"])

    # ---- matrix (SLURM) ----------------------------------------------------
    if a == "matrix":
        fws = " ".join(model.matrix_frameworks)
        exps = " ".join(model.matrix_experiments)
        incl = "INCLUDE_ABLATION=1 " if model.include_ablation else ""
        env = ('ARCH=%s EPOCHS=%s CONDA_ENV=%s FRAMEWORKS=%s EXPERIMENTS=%s %s' %
               (arch, epochs, q(conda), q(fws), q(exps), incl))
        return base_plan(commands=[env + "bash pred_ssl/scripts/launch_matrix.sh"],
                         background=False, log="", overlay_path="", overlay_dict={},
                         preflight_mode="matrix",
                         notes=["submits sbatch jobs (pretrain + afterok eval) per combo; "
                                "fill the #SBATCH <PARTITION>/<ACCOUNT> placeholders first",
                                "overlay/edited knobs are NOT forwarded to SLURM jobs in v1 "
                                "(matrix uses committed configs)",
                                "collect with the 'extract results' action after jobs finish"])

    # ---- one-shot utilities (inline) --------------------------------------
    if a == "make_subset":
        cmd = ("python pred_ssl/scripts/make_pilot_subset.py --src %s --dst ./pred_ssl/pilot_in100 "
               "--n-classes %s --n-per-class %s --splits train val" %
               (q(in100), model.value("pilot_classes"), model.value("pilot_per_class")))
        return base_plan(commands=[cmd], background=False, overlay_path="", overlay_dict={})

    if a == "gate_check":
        logf = model.gate_log or "./pred_ssl/logs/pilot_%s.log" % fw
        flags = _eval_flags(model, ["gate_leak", "gate_stuck", "gate_min_learning",
                                     "gate_min_ssl_drop"])
        cmd = ("python pred_ssl/scripts/check_pilot_gate.py %s %s" % (q(logf), " ".join(flags))).rstrip()
        return base_plan(commands=[cmd], background=False, log=logf,
                         overlay_path="", overlay_dict={})

    if a == "extract":
        cmd = ("python pred_ssl/scripts/extract_results.py --logs-dir ./pred_ssl/logs "
               "--out ./pred_ssl/results.csv")
        return base_plan(commands=[cmd], background=False, overlay_path="", overlay_dict={})

    if a == "tests":
        return base_plan(commands=["python -m pytest pred_ssl/tests/ -q"], background=False,
                         overlay_path="", overlay_dict={}, preflight_mode="test")

    raise ValueError("unknown action: %s" % a)
