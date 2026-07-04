"""
Linear probe for pred_ssl checkpoints. Adapts SimCLR-Imagenet/main_lincls.py.

Works for ImageNet-100 and CUB-200 (via --data), and a 4-way rotation probe
(--eval-rotation). The backbone is frozen; only a linear head is trained. The only
change vs the original is the checkpoint loader (reads backbone_state_dict via
pred_ssl.eval.common.load_backbone). Logging strings are unchanged so
scripts/extract_results.py and the existing parsers still match.

    python -m pred_ssl.eval.linear_probe --data ./pred_ssl/datasets/imagenet100 \
        --pretrained ./pred_ssl/checkpoints/simclr_relpred/checkpoint_0500.pth.tar
    python -m pred_ssl.eval.linear_probe --data ./pred_ssl/datasets/imagenet100 --pretrained <ckpt> --eval-rotation
    python -m pred_ssl.eval.linear_probe --data ./pred_ssl/datasets/cub200_prepared --pretrained <ckpt>
"""

import argparse
import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pred_ssl.eval.common import (AverageMeter, accuracy, build_resnet,  # noqa: E402
                                get_device, load_backbone, resolve_arch)


class RotationDataset(torch.utils.data.Dataset):
    """Each image -> 4 rotated copies (0/90/180/270); label is the rotation class."""

    ANGLES = [0, 90, 180, 270]

    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset) * 4

    def __getitem__(self, index):
        img_index = index // 4
        rot = index % 4
        img_path, _ = self.base_dataset.samples[img_index]
        img = Image.open(img_path).convert("RGB")
        if self.ANGLES[rot] != 0:
            img = img.rotate(self.ANGLES[rot])
        if self.transform is not None:
            img = self.transform(img)
        return img, rot


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for m in args.schedule:
        if epoch >= m:
            lr *= 0.1
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def run_epoch(loader, model, criterion, optimizer, device, train):
    losses, top1 = AverageMeter(), AverageMeter()
    model.eval()  # backbone frozen; head trained
    if train:
        model.fc.train()
    ctx = torch.enable_grad() if train else torch.no_grad()
    top5 = AverageMeter()
    with ctx:
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            k = min(5, output.size(1))
            acc1, acc5 = accuracy(output, target, topk=(1, k))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return losses.avg, top1.avg, top5.avg


def main():
    ap = argparse.ArgumentParser(description="pred_ssl linear probe")
    ap.add_argument("--data", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--eval-rotation", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=30.0)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--schedule", nargs="+", type=int, default=[120, 160])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    if not os.path.isdir(valdir):
        valdir = os.path.join(args.data, "test")
    object_classes = len([d for d in os.listdir(traindir)
                          if os.path.isdir(os.path.join(traindir, d))])
    num_classes = 4 if args.eval_rotation else object_classes
    task = "Rotation Classification (4 classes)" if args.eval_rotation \
        else f"Object Classification ({object_classes} classes)"

    print("=" * 70)
    print(f"Linear Evaluation: {task}")
    print(f"  pretrained: {args.pretrained}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    device = get_device()

    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    arch = resolve_arch(ckpt, args.arch)
    model = build_resnet(arch, num_classes)
    load_backbone(model, args.pretrained)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(), normalize])
    test_t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                 transforms.ToTensor(), normalize])

    if args.eval_rotation:
        train_ds = RotationDataset(datasets.ImageFolder(traindir), transform=train_t)
        val_ds = RotationDataset(datasets.ImageFolder(valdir), transform=test_t)
    else:
        train_ds = datasets.ImageFolder(traindir, train_t)
        val_ds = datasets.ImageFolder(valdir, test_t)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers, pin_memory=True)

    best_acc1 = 0.0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        tr_loss, tr_acc1, _ = run_epoch(train_loader, model, criterion, optimizer, device, train=True)
        val_loss, val_acc1, val_acc5 = run_epoch(val_loader, model, criterion, optimizer, device, train=False)
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        print(f"Epoch [{epoch + 1}/{args.epochs}]  "
              f"Train Loss: {tr_loss:.4f}  Train Acc@1: {tr_acc1:.2f}%  "
              f"Val Loss: {val_loss:.4f}  Val Acc@1: {val_acc1:.2f}%  Val Acc@5: {val_acc5:.2f}%"
              + ("  *BEST*" if is_best else ""), flush=True)

    print("\n" + "=" * 70)
    print(f"FINAL RESULTS — {task}")
    print(f"  Best Val Acc@1: {best_acc1:.2f}%")
    print(f"  Checkpoint: {args.pretrained}")
    print("=" * 70)


if __name__ == "__main__":
    main()
