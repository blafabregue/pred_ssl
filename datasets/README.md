# pred_ssl/datasets/

All datasets for `pred_ssl` live here, **inside the `pred_ssl/` folder**, so pulling just
`pred_ssl/` from GitHub gives you the whole project structure. Training and eval read
from these three subfolders (override the paths in relctl's **Runtime** group or via
the `IN100` / `CUB` / `FLOWERS` env vars):

```
pred_ssl/datasets/
├── imagenet100/          { train/  val/ }    # pretraining + IN-100 object/rotation eval
├── cub200_prepared/      { train/  val/ }    # CUB-200 linear eval
└── flowers102_prepared/  { train/  test/ }   # Flowers-102 few-shot eval
```

Each is a standard ImageFolder tree: `<split>/<class>/<images>`.

> Run all commands from the folder that *contains* `pred_ssl/` — paths are written as
> `./pred_ssl/datasets/...`.

## How to populate it

**From the Google Drive archives** (see `pred_ssl/HANDOFF.md` Section 3):
```bash
# from the folder that contains pred_ssl/
tar xf imagenet100.tar         -C pred_ssl/datasets
tar xf cub200_prepared.tar     -C pred_ssl/datasets
tar xf flowers102_prepared.tar -C pred_ssl/datasets
```

**Or symlink existing copies into here:**
```bash
ln -sfn /path/to/imagenet100         pred_ssl/datasets/imagenet100
ln -sfn /path/to/cub200_prepared     pred_ssl/datasets/cub200_prepared
ln -sfn /path/to/flowers102_prepared pred_ssl/datasets/flowers102_prepared
```

Verify:
```bash
for p in pred_ssl/datasets/imagenet100/train pred_ssl/datasets/cub200_prepared/train pred_ssl/datasets/flowers102_prepared/train; do
  [ -d "$p" ] && echo "OK   $p" || echo "MISSING $p"
done
```

> The actual image data is git-ignored; only this README is tracked, so the folder
> structure travels with the repo while the (huge) data does not.
