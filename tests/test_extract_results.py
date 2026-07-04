"""
Regression tests for scripts/extract_results.py parsing — in particular that
IN-100 and CUB-200 results stay separated (both eval steps print "Object
Classification", so section detection must be STEP-marker driven, not keyword).

Run:  python -m pytest pred_ssl/tests/test_extract_results.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pred_ssl.scripts.extract_results import parse_log, split_name  # noqa: E402

SAMPLE_LOG = """\
STEP 1: Pretrain (1 epochs)
Epoch [1/1]  Loss: 2.2953  SSL_Loss: 1.9289  Pred_Loss: 0.7327  Pred_Acc: 43.91%  LR: 0.004687
  PerFactor: rotation=37.5 hflip=56.2 brightness=50.0 contrast=75.0 saturation=20.0 hue=10.0 grayscale=31.2 blur=75.0
STEP 2: ImageNet-100 Object Linear Eval
Linear Evaluation: Object Classification (100 classes)
Epoch [1/1]  Train Loss: 1.3  Train Acc@1: 40.00%  Val Loss: 1.2  Val Acc@1: 62.00%  Val Acc@5: 90.00%  *BEST*
FINAL RESULTS — Object Classification (100 classes)
  Best Val Acc@1: 62.00%
STEP 3: ImageNet-100 Rotation Linear Eval
Linear Evaluation: Rotation Classification (4 classes)
Epoch [1/1]  Train Loss: 1.3  Train Acc@1: 30.00%  Val Loss: 1.3  Val Acc@1: 45.00%  Val Acc@5: 100.00%  *BEST*
STEP 4: CUB-200 Object Linear Eval
Linear Evaluation: Object Classification (200 classes)
Epoch [1/1]  Train Loss: 1.3  Train Acc@1: 35.00%  Val Loss: 1.25  Val Acc@1: 51.00%  Val Acc@5: 80.00%  *BEST*
STEP 5: Flowers-102 Few-shot Eval
  5-shot: 39.2% (± 4.7%)
  10-shot: 35.0% (± 1.8%)
"""


def test_parse_separates_in100_and_cub200(tmp_path):
    p = tmp_path / "simclr_relpred.log"
    p.write_text(SAMPLE_LOG)
    r = parse_log(str(p))
    assert r["in100_acc1"] == "62.00" and r["in100_acc5"] == "90.00"
    assert r["rotation_acc1"] == "45.00"
    assert r["cub200_acc1"] == "51.00" and r["cub200_acc5"] == "80.00"  # NOT overwritten by in100


def test_parse_pretrain_and_perfactor(tmp_path):
    p = tmp_path / "moco_relpred.log"
    p.write_text(SAMPLE_LOG)
    r = parse_log(str(p))
    assert r["pretrain_loss"] == "2.2953"
    assert r["pretrain_ssl_loss"] == "1.9289"
    assert r["pretrain_pred_acc"] == "43.91"
    assert r["pf_rotation"] == "37.5" and r["pf_hue"] == "10.0"


def test_parse_fewshot(tmp_path):
    p = tmp_path / "byol_baseline.log"
    p.write_text(SAMPLE_LOG)
    r = parse_log(str(p))
    assert r["flowers_5shot"] == "39.2" and r["flowers_5shot_ci"] == "4.7"
    assert r["flowers_10shot"] == "35.0" and r["flowers_10shot_ci"] == "1.8"


def test_split_name():
    assert split_name("simclr_relpred") == ("simclr", "relpred")
    assert split_name("moco_baseline") == ("moco", "baseline")
    assert split_name("simclr_relpred_lambda0") == ("simclr", "relpred_lambda0")
    assert split_name("looc_relpred") == ("looc", "relpred")


if __name__ == "__main__":
    import tempfile, pathlib  # noqa: E401
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                fn(pathlib.Path(tempfile.mkdtemp()))
            else:
                fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
