"""
Training script for SimpleDetector baseline.

Usage:
    python train.py --data_root /path/to/dataset --num_classes 10 --epochs 50
"""

import argparse
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import SimpleDetector, detection_loss
from dataloader import build_dataloader, albumentations_transform


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Simple Object Detection Baseline")
    p.add_argument("--data_root",   required=True,  help="Dataset root (contains annotations/)")
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--img_size",    type=int, default=320)
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--lr",          type=float, default=8e-4)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--resume",      default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = total_cls = total_reg = 0.0
    t0 = time.time()

    for step, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        cls_logits, bbox_preds, anchors = model(images)
        loss_cls, loss_reg = detection_loss(cls_logits, bbox_preds, anchors, targets)
        loss = loss_cls + loss_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls  += loss_cls.item()
        total_reg  += loss_reg.item()

        if (step + 1) % 20 == 0:
            avg = total_loss / (step + 1)
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d} | step {step+1:4d}/{len(loader)} "
                  f"| loss {avg:.4f} (cls {total_cls/(step+1):.4f} "
                  f"reg {total_reg/(step+1):.4f}) | {elapsed:.1f}s")

    n = len(loader)
    return total_loss / n, total_cls / n, total_reg / n


# ---------------------------------------------------------------------------
# Validation (loss only — for mAP use evaluate.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0

    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        cls_logits, bbox_preds, anchors = model(images)
        loss_cls, loss_reg = detection_loss(cls_logits, bbox_preds, anchors, targets)
        total_loss += (loss_cls + loss_reg).item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    train_loader = build_dataloader(
        data_root=args.data_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transforms=albumentations_transform(),
        normalize=True,
        drop_last=True,
        img_size=args.img_size,
    )
    val_loader = build_dataloader(
        data_root=args.data_root,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True,
        img_size=args.img_size,
    )

    # ---- Model ----
    model = SimpleDetector(num_classes=args.num_classes).to(device)

    # ---- Optimizer ----
    # Separate LR: backbone (lower) vs head (higher)
    backbone_params = list(model.layer1.parameters()) + \
                      list(model.layer2.parameters()) + \
                      list(model.layer3.parameters()) + \
                      list(model.extra.parameters())
    head_params = list(model.fpn.parameters()) + list(model.head.parameters())

    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- Resume ----
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resumed from epoch {ckpt['epoch']}")

    # ---- Training loop ----
    print(f"\nTraining for {args.epochs} epochs | {len(train_loader)} steps/epoch\n")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, cls_l, reg_l = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        print(f"[Epoch {epoch:3d}] train={train_loss:.4f} (cls={cls_l:.4f} reg={reg_l:.4f}) "
              f"val={val_loss:.4f}  lr={scheduler.get_last_lr()[1]:.2e}")

        # Save latest
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }
        torch.save(ckpt, save_dir / "last.pth")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, save_dir / "best.pth")
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

    print("\nDone. Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()