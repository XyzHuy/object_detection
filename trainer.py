from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataloader import albumentations_transform, build_dataloader
from loss import YOLOv8Loss
from metrics import detection_metrics, non_max_suppression
from model import YOLOv8Scratch


DEFAULT_IMG_SIZE = 512
DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 2e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_FREEZE_BACKBONE_EPOCHS = 5
DEFAULT_CONF_THRESHOLD = 0.001
DEFAULT_NMS_IOU = 0.65


def load_classes(data_root: str | Path, split: str = "train") -> list[str]:
    annotation_path = Path(data_root) / "annotations" / f"{split}.json"
    with annotation_path.open("r", encoding="utf-8") as file:
        return json.load(file)["classes"]


def move_targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    moved = []
    for target in targets:
        item = dict(target)
        item["boxes"] = target["boxes"].to(device)
        item["labels"] = target["labels"].to(device)
        moved.append(item)
    return moved


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    backbone = getattr(model, "backbone", None)
    if hasattr(backbone, "set_feature_extractor_trainable"):
        backbone.set_feature_extractor_trainable(trainable)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    amp: bool,
    epoch: int,
) -> dict:
    model.train()
    running = {"loss": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0}
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            outputs = model(images)
            loss_items = criterion(outputs, targets)

        scaler.scale(loss_items.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        running["loss"] += float(loss_items.loss.detach()) * batch_size
        running["box"] += float(loss_items.box_loss) * batch_size
        running["cls"] += float(loss_items.cls_loss) * batch_size
        running["dfl"] += float(loss_items.dfl_loss) * batch_size
        seen = max(progress.n + 1, 1) * batch_size
        progress.set_postfix({key: value / seen for key, value in running.items()})

    denom = len(loader.dataset)
    return {key: value / max(denom, 1) for key, value in running.items()}


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    criterion,
    device,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
) -> dict:
    model.eval()
    losses = {"loss": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0}
    all_predictions = []
    all_targets = []

    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        device_targets = move_targets_to_device(targets, device)

        outputs = model.loss_outputs(images)
        loss_items = criterion(outputs, device_targets)
        batch_size = images.shape[0]
        losses["loss"] += float(loss_items.loss.detach()) * batch_size
        losses["box"] += float(loss_items.box_loss) * batch_size
        losses["cls"] += float(loss_items.cls_loss) * batch_size
        losses["dfl"] += float(loss_items.dfl_loss) * batch_size

        model.eval()
        pred, _ = model(images)
        all_predictions.extend(
            non_max_suppression(pred, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        )
        all_targets.extend([{**target, "boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()} for target in targets])

    denom = len(loader.dataset)
    loss_metrics = {f"val_{key}": value / max(denom, 1) for key, value in losses.items()}
    det_metrics = detection_metrics(
        all_predictions,
        all_targets,
        num_classes=model.head.num_classes,
        iou_threshold=0.5,
    )
    return {**loss_metrics, **det_metrics}


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, classes: list[str], metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "classes": classes,
            "metrics": metrics,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train YOLOv8 with ImageNet-pretrained CSPDarkNet backbone")
    parser.add_argument("--data_root", default="final_public/public")
    # Dataset-informed defaults:
    # - Images are mostly <= 500px on the long edge, so 512 keeps detail without
    #   excessive upscaling.
    # - Small/tiny objects are common, especially person/car/chair, so 320 is
    #   too lossy for this dataset.
    # - CSPDarkNet53 is heavier than the previous scratch backbone; batch 4 is
    #   a safer default, paired with a conservative LR.
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--conf_threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--nms_iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=DEFAULT_FREEZE_BACKBONE_EPOCHS)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--scratch_backbone", action="store_true")
    parser.add_argument("--resume", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(args.data_root)

    transforms = None if args.no_aug else albumentations_transform()
    train_loader = build_dataloader(
        args.data_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transforms=transforms,
        img_size=args.img_size,
    )
    val_loader = build_dataloader(
        args.data_root,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transforms=None,
        img_size=args.img_size,
    )

    model = YOLOv8Scratch(
        num_classes=len(classes),
        pretrained_backbone=not args.scratch_backbone,
        use_cspdarknet=not args.scratch_backbone,
    ).to(device)
    criterion = YOLOv8Loss(num_classes=len(classes), strides=model.head.strides, reg_max=model.head.reg_max)

    set_backbone_trainable(model, args.freeze_backbone_epochs <= 0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))
    start_epoch = 1
    best_map = -1.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_map = float(checkpoint.get("metrics", {}).get("mAP50", -1.0))

    output_dir = Path(args.output_dir)
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, device.type == "cuda" and not args.no_amp, epoch
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.nms_iou,
        )
        scheduler.step()

        metrics = {**train_metrics, **val_metrics, "lr": scheduler.get_last_lr()[0]}
        print(
            f"epoch {epoch:03d}: loss={metrics['loss']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} mAP50={metrics['mAP50']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}"
        )
        save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler, epoch, classes, metrics)
        if metrics["mAP50"] > best_map:
            best_map = metrics["mAP50"]
            save_checkpoint(output_dir / "best.pth", model, optimizer, scheduler, epoch, classes, metrics)


if __name__ == "__main__":
    main()
