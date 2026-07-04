"""relctl — interactive control panel for the pred_ssl SSL repo.

Run it from the repo root inside the project's conda env:

    python -m pred_ssl.relctl              # auto: Rich tier if installed, else plain
    python -m pred_ssl.relctl --plain      # force the zero-dependency plain tier
    python -m pred_ssl.relctl --validate   # check the knob catalog against the configs
"""

__all__ = ["app", "config", "actions", "jobs", "knobs", "preflight", "ui"]
