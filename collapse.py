"""
Collapse diagnostics for the frozen representation.

Positives-only pretraining (models/frameworks/posonly.py) can fail in two ways that
a loss curve hides completely: the loss goes DOWN in both. So the run has to be
scored on the geometry of the features themselves, not on the objective.

Two numbers, both computed on L2-normalized features:

``std_ratio`` -- the per-dimension standard deviation averaged over dimensions,
    rescaled by sqrt(D). SimSiam's collapse metric. Features spread over the unit
    sphere have per-dimension std ~ 1/sqrt(D), so the ratio sits near 1; a constant
    representation drives it to 0. Catches COMPLETE collapse.

``erank`` -- effective rank (Roy & Vetterli, 2007): exp of the Shannon entropy of
    the covariance eigenspectrum normalized to a distribution. Reads as "how many
    dimensions are really in use": D for an isotropic spectrum, 1 when all the
    variance is on one direction. Catches DIMENSIONAL collapse, which is the
    failure mode that actually threatens this method -- a representation that keeps
    only the few directions needed to predict the augmentation parameters and drops
    the image content scores a healthy std_ratio while erank falls to single digits.

READ THE TWO TOGETHER; neither is sufficient alone, in either direction:

  - std_ratio passes a representation that encodes nothing but omega, since a
    3-dimensional code spread over the sphere has a perfectly healthy
    per-dimension spread. That is the case erank exists to catch.
  - erank is SCALE-INVARIANT by construction -- it describes the shape of the
    spectrum, not its size. Features that have collapsed to a point plus isotropic
    numerical noise have a flat spectrum and therefore a HIGH effective rank. That
    is the case std_ratio exists to catch.

So a healthy run is one where both are high, and which of the two dropped tells
you which failure occurred.
"""

import torch

# Cost guard: erank needs a D x D covariance, so its cost is set by D, but the
# accumulation is linear in N and there is no accuracy to gain from a huge sample
# once N comfortably exceeds D.
MAX_SAMPLES = 16384


@torch.no_grad()
def collapse_stats(feats, max_samples=MAX_SAMPLES):
    """Collapse diagnostics for an (N, D) matrix of L2-NORMALIZED features.

    Returns {"std", "std_ratio", "erank", "erank_ratio"}. erank is bounded by
    min(N - 1, D), so erank_ratio divides by that bound rather than by D -- with a
    small feature bank the raw number would otherwise look artificially collapsed.
    """
    x = feats.detach().float()
    n, d = x.shape
    if n > max_samples:
        x = x[torch.randperm(n, device=x.device)[:max_samples]]
        n = x.shape[0]

    # Guard first: a single sample makes std() NaN (zero degrees of freedom), and a
    # NaN here would propagate silently into the log line and the curves.
    if n < 2:
        return {"std": 0.0, "std_ratio": 0.0, "erank": 1.0, "erank_ratio": 0.0}

    std = x.std(dim=0).mean().item()
    std_ratio = std * (d ** 0.5)

    xc = x - x.mean(dim=0, keepdim=True)
    cov = (xc.t() @ xc).double() / (n - 1)
    lam = torch.linalg.eigvalsh(cov).clamp_min(0)
    total = lam.sum()
    if total <= 0:
        return {"std": std, "std_ratio": std_ratio, "erank": 1.0, "erank_ratio": 0.0}
    p = lam / total
    p = p[p > 1e-12]
    erank = torch.exp(-(p * p.log()).sum()).item()
    return {"std": std, "std_ratio": std_ratio, "erank": erank,
            "erank_ratio": erank / min(n - 1, d)}


def format_stats(stats):
    """The log line, in the KNN_Acc style so plot_curves can pick it up."""
    return (f"Collapse: std_ratio {stats['std_ratio']:.4f}  "
            f"erank {stats['erank']:.1f}  erank_ratio {stats['erank_ratio']:.4f}")
