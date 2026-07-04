"""
Latent-space partition for the disentanglement experiment (relpred_split).

When ``cfg["feat_split"]`` is true, the backbone feature h (post-avgpool) is split
into three CONTIGUOUS blocks ``[ vanilla | common | rel ]``:

  - the SSL projection head consumes  h[:, : n_vanilla + n_common]   (vanilla + common)
  - the relational head consumes      h[:, n_vanilla :]              (common + rel)

so the vanilla-exclusive block never receives the relational gradient, the
rel-exclusive block never receives the contrastive gradient, and the common block
is pulled by both — a gradient-level factorization of the representation. The trunk
below h stays fully shared, so the separation says who OPTIMIZES each block, not
what each block CONTAINS; ``split_decov_lambda`` (losses.SplitDecovLoss) optionally
pushes the two exclusive blocks towards actual decorrelation.

Ratios come from ``cfg["split_ratios"] = [vanilla, common, rel]`` (non-negative,
sum 1; default [0.5, 0.25, 0.25]). Disabled (the default) both heads see the full
h and every framework behaves exactly as before.
"""

DEFAULT_RATIOS = (0.5, 0.25, 0.25)

# Named sub-spaces for evaluation (eval/linear_probe.py --feat-slice).
PARTS = ("full", "vanilla", "common", "rel", "ssl", "relhead")


class FeatSplit:
    """Integer block sizes + slicing helpers for one feat_dim."""

    def __init__(self, feat_dim, ratios=DEFAULT_RATIOS, enabled=False):
        self.feat_dim = feat_dim
        self.enabled = bool(enabled)
        r = list(ratios if ratios is not None else DEFAULT_RATIOS)
        if len(r) != 3 or any(x < 0 for x in r) or abs(sum(r) - 1.0) > 1e-6:
            raise ValueError(
                "split_ratios must be 3 non-negative fractions [vanilla, common, rel] "
                f"summing to 1, got {r!r}")
        self.n_common = int(round(r[1] * feat_dim))
        self.n_rel = int(round(r[2] * feat_dim))
        self.n_vanilla = feat_dim - self.n_common - self.n_rel
        if self.enabled:
            if self.n_vanilla + self.n_common == 0:
                raise ValueError("split leaves the SSL head with 0 input dims")
            if self.n_common + self.n_rel == 0:
                raise ValueError("split leaves the relational head with 0 input dims")
        # Input widths of the two heads (full feat_dim when the split is disabled).
        self.ssl_dim = (self.n_vanilla + self.n_common) if self.enabled else feat_dim
        self.rel_dim = (self.n_common + self.n_rel) if self.enabled else feat_dim

    # ---- training-time slices -------------------------------------------------
    def ssl(self, h):
        """The slice the SSL projection head consumes (vanilla + common)."""
        return h[:, : self.ssl_dim] if self.enabled else h

    def rel(self, h):
        """The slice the relational head consumes (common + rel)."""
        return h[:, self.n_vanilla:] if self.enabled else h

    def vanilla_excl(self, h):
        """The vanilla-EXCLUSIVE block (for the decorrelation penalty)."""
        return h[:, : self.n_vanilla]

    def rel_excl(self, h):
        """The rel-EXCLUSIVE block (for the decorrelation penalty)."""
        return h[:, self.n_vanilla + self.n_common:]

    # ---- evaluation-time bounds -----------------------------------------------
    def bounds(self, part):
        """(start, end) of a named sub-space of h — used by the per-slice probe."""
        v, c = self.n_vanilla, self.n_common
        table = {
            "full": (0, self.feat_dim),
            "vanilla": (0, v),
            "common": (v, v + c),
            "rel": (v + c, self.feat_dim),
            "ssl": (0, v + c),          # what the SSL head saw
            "relhead": (v, self.feat_dim),  # what the relational head saw
        }
        if part not in table:
            raise ValueError(f"unknown slice '{part}' (choices: {', '.join(PARTS)})")
        return table[part]


def build_split(cfg, feat_dim):
    return FeatSplit(feat_dim,
                     ratios=cfg.get("split_ratios", DEFAULT_RATIOS),
                     enabled=cfg.get("feat_split", False))
