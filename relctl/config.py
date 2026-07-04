"""
ConfigModel — the live state behind the relctl control panel.

Train-domain defaults are read from configs/*.yaml at runtime and merged exactly as
train.py does (base <- framework <- experiment), so the values relctl shows can never
drift from the committed configs. User edits are layered on top; on launch they are
split into train.py CLI flags (knobs with a `cli_flag`) and a generated YAML overlay
(YAML-only / framework-specific knobs), mirroring train.py's --config-overlay hook.
"""

import os
import copy

import yaml

from .knobs import (KNOBS, KNOBS_BY_KEY, FRAMEWORK_KNOBS, FRAMEWORKS, EXPERIMENTS,
                    DELTA_KEYS)


def _deep_merge(a, b):
    """Recursive dict merge (same semantics as train.py._deep_merge)."""
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def _load_yaml(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


class ValidationError(ValueError):
    pass


class ConfigModel:
    def __init__(self, repo_root):
        self.repo_root = repo_root
        self.config_dir = os.path.join(repo_root, "pred_ssl", "configs")
        self.framework = "simclr"
        self.experiment = "relpred"
        self.edits = {}            # key -> value (user overrides, any domain)
        self.action = "pilot"
        self.eval_ckpt = ""        # for eval-only / single-eval actions
        self.resume_ckpt = ""      # for resume
        self.gate_log = ""         # log to run the pilot gate against
        self.matrix_frameworks = list(FRAMEWORKS)
        self.matrix_experiments = ["baseline", "relpred"]
        self.include_ablation = False  # add relpred_lambda0 to the matrix
        self.profile_name = ""     # last loaded/saved profile, "" = unsaved
        self._cache = {}           # (fw, exp) -> merged base config

    # ------------------------------------------------------------------ merge
    def _merged_base(self):
        """base.yaml <- framework/<fw>.yaml <- experiment/<exp>.yaml (no user edits)."""
        key = (self.framework, self.experiment)
        if key not in self._cache:
            cfg = _load_yaml(os.path.join(self.config_dir, "base.yaml"))
            _deep_merge(cfg, _load_yaml(os.path.join(self.config_dir, "framework",
                                                     self.framework + ".yaml")))
            if self.experiment:
                _deep_merge(cfg, _load_yaml(os.path.join(self.config_dir, "experiment",
                                                         self.experiment + ".yaml")))
            cfg["framework"] = self.framework
            self._cache[key] = cfg
        return self._cache[key]

    def resolved_train_cfg(self):
        """The effective train.py config: merged base + train-domain user edits."""
        cfg = copy.deepcopy(self._merged_base())
        for k, v in self.edits.items():
            kn = KNOBS_BY_KEY.get(k)
            if kn is not None and kn.domain == "train":
                cfg[k] = copy.deepcopy(v)
        return cfg

    def base_lr(self):
        cfg = self.resolved_train_cfg()
        scale = (cfg.get("batch_size", 256) / 256.0) if cfg.get("lr_scale_by_batch") else 1.0
        return cfg.get("lr", 0.3) * scale

    # ------------------------------------------------------------------ values
    def baseline(self, key):
        """The value a knob would have with NO user edit."""
        kn = KNOBS_BY_KEY[key]
        if kn.domain == "train":
            return self._merged_base().get(key, kn.default)
        return kn.default

    def value(self, key):
        if key in self.edits:
            return self.edits[key]
        return self.baseline(key)

    def is_dirty(self, key):
        return key in self.edits and self.edits[key] != self.baseline(key)

    def dirty_keys(self):
        return [k for k in self.edits if self.is_dirty(k)]

    # ------------------------------------------------------------------ edits
    def set(self, key, raw):
        """Validate and stage an edit. Returns the parsed value or raises ValidationError."""
        kn = KNOBS_BY_KEY[key]
        val = self._validate(kn, raw)
        self.edits[key] = val
        return val

    def set_delta_key(self, subkey, raw):
        kn = KNOBS_BY_KEY["delta"]
        try:
            v = float(raw)
        except (TypeError, ValueError):
            raise ValidationError("delta.%s must be a number" % subkey)
        if v <= 0:
            raise ValidationError("delta.%s must be > 0 (a 0 gap makes 'different' == 'same')" % subkey)
        cur = copy.deepcopy(self.value("delta"))
        cur[subkey] = v
        self.edits["delta"] = cur

    def reset(self, key=None):
        if key is None:
            self.edits.clear()
        else:
            self.edits.pop(key, None)

    def set_framework(self, fw):
        if fw not in FRAMEWORKS:
            raise ValidationError("unknown framework: %s" % fw)
        self.framework = fw

    def set_experiment(self, exp):
        if exp not in EXPERIMENTS:
            raise ValidationError("unknown experiment: %s" % exp)
        self.experiment = exp

    # -------------------------------------------------------------- validation
    def _validate(self, kn, raw):
        t = kn.type
        if t == "int":
            v = self._as_int(kn, raw)
        elif t == "float":
            v = self._as_float(kn, raw)
        elif t == "bool":
            v = self._as_bool(raw)
        elif t == "enum":
            v = str(raw).strip()
            if v not in kn.valid:
                raise ValidationError("%s must be one of %s" % (kn.key, kn.valid))
        elif t in ("path", "str"):
            v = str(raw).strip()
            if not v:
                raise ValidationError("%s must not be empty" % kn.key)
        elif t == "list_int":
            v = [self._num(x, int, kn) for x in self._split(raw)]
            self._check_list(kn, v)
        elif t == "list_float":
            v = [self._num(x, float, kn) for x in self._split(raw)]
            self._check_list(kn, v)
        elif t == "list_str":
            v = self._split(raw, cast=str)
        elif t == "dict_float":
            raise ValidationError("edit delta sub-keys individually")
        else:
            raise ValidationError("unhandled type %s" % t)

        # framework guards
        if kn.key == "n_aug" and v != 0:
            raise ValidationError("looc v1 supports n_aug=0 only (n_aug!=0 raises "
                                  "NotImplementedError at train start)")
        if kn.key == "full_multiview" and v:
            raise ValidationError("looc v1 supports full_multiview=false only (true raises "
                                  "NotImplementedError at train start)")
        return v

    def _as_int(self, kn, raw):
        try:
            v = int(str(raw).strip())
        except ValueError:
            raise ValidationError("%s must be an integer" % kn.key)
        self._range(kn, v)
        return v

    def _as_float(self, kn, raw):
        try:
            v = float(str(raw).strip())
        except ValueError:
            raise ValidationError("%s must be a number" % kn.key)
        self._range(kn, v)
        return v

    @staticmethod
    def _as_bool(raw):
        s = str(raw).strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
        raise ValidationError("expected a boolean (true/false)")

    def _range(self, kn, v):
        if not isinstance(kn.valid, tuple):
            return
        lo, hi = kn.valid
        if lo is not None and v < lo:
            raise ValidationError("%s must be >= %s" % (kn.key, lo))
        if hi is not None and v > hi:
            raise ValidationError("%s must be <= %s" % (kn.key, hi))

    def _num(self, x, cast, kn):
        try:
            v = cast(x)
        except ValueError:
            raise ValidationError("%s: '%s' is not a %s" % (kn.key, x, cast.__name__))
        self._range(kn, v)
        return v

    @staticmethod
    def _split(raw, cast=None):
        if isinstance(raw, (list, tuple)):
            items = list(raw)
        else:
            items = [p for p in str(raw).replace(",", " ").split() if p]
        return [cast(i) for i in items] if cast else items

    def _check_list(self, kn, v):
        if kn.key == "crop_scale":
            if len(v) != 2 or not (0.0 <= v[0] < v[1] <= 1.0):
                raise ValidationError("crop_scale must be two values 0 <= lo < hi <= 1.0")
        if kn.key == "split_ratios":
            if len(v) != 3 or any(x < 0 for x in v) or abs(sum(v) - 1.0) > 1e-6:
                raise ValidationError("split_ratios must be 3 non-negative values "
                                      "[vanilla, common, rel] summing to 1")
            if v[0] + v[1] <= 0 or v[1] + v[2] <= 0:
                raise ValidationError("split_ratios leaves a head with 0 input dims "
                                      "(vanilla+common and common+rel must both be > 0)")
        if kn.key in ("schedule", "lincls_schedule") and v != sorted(v):
            raise ValidationError("%s milestones must be ascending" % kn.key)
        if kn.key == "n_shots" and (not v or any(x < 1 for x in v)):
            raise ValidationError("n_shots must be positive integers")

    # -------------------------------------------------------------- couplings
    def warnings(self):
        """Advisory coupling/foot-gun checks (non-blocking)."""
        out = []
        cfg = self.resolved_train_cfg()
        rl = cfg.get("rel_lambda", 0.0)
        sharing = cfg.get("aug_sharing", False)
        if rl > 0 and not sharing:
            out.append(("err", "rel_lambda=%.3g but aug_sharing=OFF -> the head gets all-zero "
                               "labels (meaningless). Turn sharing ON or set rel_lambda=0." % rl))
        if rl == 0 and sharing:
            out.append(("info", "aug_sharing=ON with rel_lambda=0 -> sharing-loader ablation "
                               "(relpred_lambda0 style); not a pure-SSL control."))
        if cfg.get("lr_schedule") == "cosine" and self.is_dirty("schedule"):
            out.append(("info", "schedule milestones are ignored under lr_schedule=cosine."))
        if cfg.get("lr_schedule") == "step" and not cfg.get("schedule"):
            out.append(("err", "lr_schedule=step but `schedule` is empty -> LR never decays."))
        # experiment override notice
        for k in ("rel_lambda", "aug_sharing"):
            if self.is_dirty(k):
                exp_val = self._merged_base().get(k)
                out.append(("info", "%s manually set to %s (overrides experiment '%s' value %s)."
                            % (k, self.value(k), self.experiment, exp_val)))
        if self.framework in ("moco", "looc") and rl > 0:
            out.append(("info", "rel_lambda>0 on %s adds a second view forward (~1.5x backbone "
                               "cost)." % self.framework))
        if cfg.get("feat_split", False) and rl == 0:
            out.append(("err", "feat_split=ON but rel_lambda=0 -> no relational head, so the "
                               "rel-exclusive block gets NO gradient at all (dead dims)."))
        if cfg.get("split_decov_lambda", 0.0) > 0 and not cfg.get("feat_split", False):
            out.append(("info", "split_decov_lambda>0 is ignored while feat_split=OFF."))
        return out

    # ---------------------------------------------------------------- overlay
    # arch/epochs/data/save_dir are ALWAYS passed as explicit CLI flags (by both the
    # direct train.py call and the wrapper scripts), so they never go in the overlay.
    # Everything else dirty + train-domain rides the overlay — uniform across the
    # direct and wrapper launch paths (the wrappers only forward CONFIG_OVERLAY plus
    # those four flags, so this is the one path that carries every other knob).
    IDENTITY_FLAGS = ("arch", "epochs", "data", "save_dir")

    def overlay_dict(self):
        """Every dirty train-domain knob (except the identity flags) -> overlay YAML.

        Framework-specific knobs are included only for the ACTIVE framework, so a
        leftover `temperature` edit never leaks into a moco overlay."""
        out = {}
        for k in self.dirty_keys():
            kn = KNOBS_BY_KEY[k]
            if kn.domain != "train" or k in self.IDENTITY_FLAGS:
                continue
            if kn.fw_scope is not None and kn.fw_scope != self.framework:
                continue
            out[k] = copy.deepcopy(self.value(k))
        return out

    def needs_overlay(self):
        return bool(self.overlay_dict())

    def write_overlay(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.overlay_dict(), f, default_flow_style=False, sort_keys=True)
        return path

    # ---------------------------------------------------------------- profiles
    def profiles_dir(self):
        return os.path.join(self.repo_root, "pred_ssl", "relctl", "profiles")

    def save_profile(self, name):
        d = self.profiles_dir()
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name + ".yaml")
        with open(path, "w") as f:
            yaml.safe_dump({
                "framework": self.framework,
                "experiment": self.experiment,
                "action": self.action,
                "eval_ckpt": self.eval_ckpt,
                "resume_ckpt": self.resume_ckpt,
                "edits": self.edits,
            }, f, default_flow_style=False, sort_keys=True)
        self.profile_name = name
        return path

    def load_profile(self, name):
        path = os.path.join(self.profiles_dir(), name + ".yaml")
        data = _load_yaml(path)
        if not data:
            raise ValidationError("profile not found or empty: %s" % path)
        self.framework = data.get("framework", self.framework)
        self.experiment = data.get("experiment", self.experiment)
        self.action = data.get("action", self.action)
        self.eval_ckpt = data.get("eval_ckpt", "")
        self.resume_ckpt = data.get("resume_ckpt", "")
        self.edits = data.get("edits", {}) or {}
        self.profile_name = name
        return path

    def list_profiles(self):
        d = self.profiles_dir()
        if not os.path.isdir(d):
            return []
        return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".yaml"))
