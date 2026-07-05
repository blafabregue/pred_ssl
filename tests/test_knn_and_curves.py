"""
Tests for the pretraining kNN monitor (eval/knn.py), the best/last checkpoint
plumbing pieces, and the curves parser (scripts/plot_curves.py).

Run:  python -m pytest pred_ssl/tests/test_knn_and_curves.py -q
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.eval.knn import build_knn_monitor, knn_predict  # noqa: E402
from pred_ssl.models.frameworks import build_model  # noqa: E402
from pred_ssl.scripts.plot_curves import parse_curves, write_csv  # noqa: E402


# ---------------------------------------------------------------------------
# (1) knn_predict on synthetic, well-separated clusters
# ---------------------------------------------------------------------------

def test_knn_predict_separable_clusters():
    g = torch.Generator().manual_seed(0)
    c0 = F.normalize(torch.randn(20, 8, generator=g) * 0.05 + torch.tensor([5.0] + [0.0] * 7), dim=1)
    c1 = F.normalize(torch.randn(20, 8, generator=g) * 0.05 + torch.tensor([0.0] * 7 + [5.0]), dim=1)
    bank = torch.cat([c0, c1])
    labels = torch.tensor([0] * 20 + [1] * 20)
    query = torch.cat([c0[:5], c1[:5]])
    pred = knn_predict(query, bank, labels, num_classes=2, k=5)
    assert torch.equal(pred, torch.tensor([0] * 5 + [1] * 5))


def test_knn_predict_k_clamped_to_bank():
    bank = F.normalize(torch.randn(3, 4), dim=1)
    pred = knn_predict(bank, bank, torch.tensor([0, 1, 0]), num_classes=2, k=50)
    assert pred.shape == (3,)


# ---------------------------------------------------------------------------
# (2) monitor end-to-end on a tiny on-disk ImageFolder
# ---------------------------------------------------------------------------

def _make_imagefolder(root, split, n_classes=2, n_per_class=4, size=64, seed=0):
    rng = np.random.RandomState(seed)
    for c in range(n_classes):
        d = os.path.join(root, split, f"class{c}")
        os.makedirs(d, exist_ok=True)
        for i in range(n_per_class):
            arr = (rng.rand(size, size, 3) * 255).astype("uint8")
            Image.fromarray(arr, "RGB").save(os.path.join(d, f"img{i}.png"))


def test_knn_monitor_end_to_end(tmp_path):
    data = str(tmp_path / "data")
    _make_imagefolder(data, "train")
    _make_imagefolder(data, "val", n_per_class=2, seed=1)
    cfg = {"data": data, "knn_eval_freq": 1, "knn_k": 3, "knn_temp": 0.07,
           "knn_per_class": 4, "batch_size": 8, "workers": 0,
           "arch": "resnet18", "framework": "simclr", "rel_lambda": 0.0,
           "simclr_dim": 32, "temperature": 0.5}
    knn = build_knn_monitor(cfg)
    assert knn is not None and knn.num_classes == 2
    model = build_model(cfg).train()
    acc = knn.evaluate(model, "simclr", "cpu")
    assert 0.0 <= acc <= 100.0
    assert model.training, "evaluate() must restore train mode"


def test_knn_monitor_off_or_missing_val(tmp_path):
    data = str(tmp_path / "data")
    _make_imagefolder(data, "train")
    assert build_knn_monitor({"data": data, "knn_eval_freq": 0}) is None   # off
    assert build_knn_monitor({"data": data, "knn_eval_freq": 5}) is None   # no val/


# ---------------------------------------------------------------------------
# (3) curves parser: pretrain + KNN + multiple eval tasks, resume overwrite
# ---------------------------------------------------------------------------

SAMPLE = """\
Epoch [1/3]  Loss: 6.84  SSL_Loss: 6.10  Pred_Loss: 1.46  Pred_Acc: 49.86%  LR: 0.3
  PerFactor: rotation=46.2 crop=49.2
Epoch [2/3]  Loss: 6.62  SSL_Loss: 5.95  Pred_Loss: 1.33  Pred_Acc: 50.06%  LR: 0.27
  KNN_Acc: 22.40%  (epoch 2)
Epoch [2/3]  Loss: 6.60  SSL_Loss: 5.94  Pred_Loss: 1.30  Pred_Acc: 50.50%  LR: 0.27
Linear Evaluation: Object Classification (100 classes)
Epoch [1/2]  Train Loss: 1.3  Train Acc@1: 40.00%  Val Loss: 1.2  Val Acc@1: 62.00%  Val Acc@5: 90.00%  *BEST*
Epoch [2/2]  Train Loss: 1.1  Train Acc@1: 45.00%  Val Loss: 1.1  Val Acc@1: 64.00%  Val Acc@5: 91.00%  *BEST*
Linear Evaluation: Rotation Classification (4 classes)
Epoch [1/2]  Train Loss: 1.3  Train Acc@1: 30.00%  Val Loss: 1.3  Val Acc@1: 45.00%  Val Acc@5: 100.00%  *BEST*
"""


def test_parse_curves(tmp_path):
    p = tmp_path / "simclr_relpred.log"
    p.write_text(SAMPLE)
    tasks = parse_curves(str(p))
    pre = tasks["pretrain"]
    assert pre[1]["loss"] == "6.84"
    assert pre[2]["loss"] == "6.60", "resumed epoch must keep the LAST occurrence"
    assert pre[2]["knn_acc"] == "22.40"
    obj = tasks["Object Classification (100 classes)"]
    assert obj[2]["val_acc1"] == "64.00" and obj[1]["val_loss"] == "1.2"
    rot = tasks["Rotation Classification (4 classes)"]
    assert rot[1]["val_acc1"] == "45.00"

    out = write_csv(tasks, str(tmp_path / "c.csv"))
    text = open(out).read()
    assert "pretrain,1,6.84" in text and "22.40" in text


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
