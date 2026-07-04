"""
Build a small ImageNet-100 subset for a fast Phase-2 pilot, using SYMLINKS only
(the source images are never copied or modified -- matches the repo's symlink
convention, e.g. `ln -sf .../imagenet100`).

    python pred_ssl/scripts/make_pilot_subset.py \
        --src ./pred_ssl/datasets/imagenet100 --dst ./pred_ssl/pilot_in100 \
        --n-classes 10 --n-per-class 100 --splits train val

The result is an ImageFolder-compatible tree of symlinks under --dst.
"""

import argparse
import os


IMG_EXTS = (".jpeg", ".jpg", ".png", ".bmp", ".webp")


def link_split(src_split, dst_split, n_classes, n_per_class):
    classes = sorted(d for d in os.listdir(src_split)
                     if os.path.isdir(os.path.join(src_split, d)))[:n_classes]
    n_imgs = 0
    for cls in classes:
        src_dir = os.path.join(src_split, cls)
        dst_dir = os.path.join(dst_split, cls)
        os.makedirs(dst_dir, exist_ok=True)
        imgs = sorted(f for f in os.listdir(src_dir)
                      if f.lower().endswith(IMG_EXTS))[:n_per_class]
        for f in imgs:
            link = os.path.join(dst_dir, f)
            if not os.path.lexists(link):
                os.symlink(os.path.abspath(os.path.join(src_dir, f)), link)
            n_imgs += 1
    return len(classes), n_imgs


def main():
    ap = argparse.ArgumentParser(description="symlink a small IN-100 pilot subset")
    ap.add_argument("--src", required=True, help="source dataset root (with train/, val/)")
    ap.add_argument("--dst", required=True, help="destination subset root (symlinks)")
    ap.add_argument("--n-classes", type=int, default=10)
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    for split in args.splits:
        src_split = os.path.join(args.src, split)
        if not os.path.isdir(src_split):
            # val/ may be named test/ for some datasets
            alt = os.path.join(args.src, "test")
            if split == "val" and os.path.isdir(alt):
                src_split = alt
            else:
                print(f"  skip '{split}': {src_split} not found")
                continue
        dst_split = os.path.join(args.dst, split)
        nc, ni = link_split(src_split, dst_split, args.n_classes, args.n_per_class)
        print(f"  {split}: {nc} classes, {ni} symlinks -> {dst_split}")
    print(f"=> pilot subset ready at {args.dst}")


if __name__ == "__main__":
    main()
