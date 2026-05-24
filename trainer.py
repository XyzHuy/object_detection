from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataloader import albumentations_transform, build_dataloader
from loss import YOLOv8Loss
from metrics import detection_metrics, filter_predictions_by_score, non_max_suppression
from model import YOLOv8Scratch


DEFAULT_IMG_SIZE = 512
DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 5e-4
DEFAULT_BACKBONE_LR = 7.5e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_FREEZE_BACKBONE_EPOCHS = 5
DEFAULT_EARLY_STOP_PATIENCE = 20
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_CONF_THRESHOLD = 0.001
DEFAULT_NMS_IOU = 0.65
DEFAULT_MAX_DET = 100
DEFAULT_LOCAL_JEPA_ALPHA = 0.2
DEFAULT_LOCAL_JEPA_FINAL_ALPHA = 0.05
DEFAULT_LOCAL_JEPA_WARMUP_EPOCHS = 5
DEFAULT_LOCAL_JEPA_DECAY_START_EPOCH = 50


def setup_logger(log_dir: str | Path) -> tuple[logging.Logger, Path]:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, log_path


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


def local_jepa_alpha_for_epoch(
    epoch: int,
    total_epochs: int,
    peak_alpha: float,
    final_alpha: float,
    warmup_epochs: int,
    decay_start_epoch: int,
) -> float:
    if peak_alpha <= 0.0:
        return 0.0

    warmup_epochs = max(int(warmup_epochs), 0)
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return peak_alpha * max(epoch - 1, 0) / warmup_epochs

    decay_start_epoch = max(int(decay_start_epoch), 1)
    if epoch < decay_start_epoch or decay_start_epoch >= total_epochs:
        return peak_alpha

    progress = (epoch - decay_start_epoch) / max(total_epochs - decay_start_epoch, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_alpha + (peak_alpha - final_alpha) * cosine


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
    running = {"loss": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0, "jepa": 0.0}
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
        running["jepa"] += float(loss_items.local_jepa_loss) * batch_size
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
    max_det: int = DEFAULT_MAX_DET,
    report_thresholds=(0.05, 0.25, 0.50),
) -> dict:
    model.eval()
    losses = {"loss": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0, "jepa": 0.0}
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
        losses["jepa"] += float(loss_items.local_jepa_loss) * batch_size

        model.eval()
        pred, _ = model(images)
        all_predictions.extend(
            non_max_suppression(pred, conf_threshold=conf_threshold, iou_threshold=iou_threshold, max_det=max_det)
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
    threshold_metrics = {}
    for threshold in report_thresholds:
        filtered_predictions = filter_predictions_by_score(all_predictions, threshold)
        metrics_at_threshold = detection_metrics(
            filtered_predictions,
            all_targets,
            num_classes=model.head.num_classes,
            iou_threshold=0.5,
        )
        suffix = f"@{threshold:g}"
        threshold_metrics[f"precision{suffix}"] = metrics_at_threshold["precision"]
        threshold_metrics[f"recall{suffix}"] = metrics_at_threshold["recall"]
        threshold_metrics[f"num_predictions{suffix}"] = metrics_at_threshold["num_predictions"]

    return {**loss_metrics, **det_metrics, **threshold_metrics}


def build_optimizer(model, lr: float, backbone_lr: float, weight_decay: float):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr, "name": "head_neck"})
    return AdamW(param_groups, weight_decay=weight_decay)


def save_checkpoint(path: Path, model, epoch: int, classes: list[str], metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compact_state = {
        key: value.detach().cpu().half() if torch.is_floating_point(value) else value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("local_jepa.")
    }
    torch.save(
        {
            "epoch": epoch,
            "model": compact_state,
            "classes": classes,
            "metrics": metrics,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train YOLOv8 with ImageNet-pretrained CSPDarkNet backbone")
    parser.add_argument("--data_root", default="final_public/public")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--backbone_lr", type=float, default=DEFAULT_BACKBONE_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--conf_threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--nms_iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--max_det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=DEFAULT_FREEZE_BACKBONE_EPOCHS)
    parser.add_argument("--early_stop_patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--min_delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--use_local_jepa", action="store_true")
    parser.add_argument("--local_jepa_alpha", type=float, default=DEFAULT_LOCAL_JEPA_ALPHA)
    parser.add_argument("--local_jepa_final_alpha", type=float, default=DEFAULT_LOCAL_JEPA_FINAL_ALPHA)
    parser.add_argument("--local_jepa_warmup_epochs", type=int, default=DEFAULT_LOCAL_JEPA_WARMUP_EPOCHS)
    parser.add_argument("--local_jepa_decay_start_epoch", type=int, default=DEFAULT_LOCAL_JEPA_DECAY_START_EPOCH)
    parser.add_argument("--no_local_jepa_alpha_schedule", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--scratch_backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger(args.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(args.data_root)

    logger.info("Starting training run")
    logger.info("Log file: %s", log_path)
    logger.info("Args: %s", json.dumps(vars(args), ensure_ascii=False, sort_keys=True))
    logger.info("Device: %s", device)
    logger.info("Classes (%d): %s", len(classes), ", ".join(classes))

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
        use_local_jepa=args.use_local_jepa,
    ).to(device)
    criterion = YOLOv8Loss(
        num_classes=len(classes),
        strides=model.head.strides,
        reg_max=model.head.reg_max,
        local_jepa_gain=0.0,
    )

    set_backbone_trainable(model, args.freeze_backbone_epochs <= 0)
    optimizer = build_optimizer(
        model,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=min(args.lr, args.backbone_lr) * 0.05,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))
    best_map = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    logger.info(
        "Optimizer groups: lr=%.6g backbone_lr=%.6g weight_decay=%.6g",
        args.lr,
        args.backbone_lr,
        args.weight_decay,
    )
    if args.use_local_jepa:
        if args.local_jepa_alpha < 0:
            raise ValueError("--local_jepa_alpha must be non-negative")
        if args.local_jepa_final_alpha < 0:
            raise ValueError("--local_jepa_final_alpha must be non-negative")
        if args.local_jepa_final_alpha > args.local_jepa_alpha:
            raise ValueError("--local_jepa_final_alpha must be <= --local_jepa_alpha")
        logger.info(
            "Local JEPA enabled: peak_alpha=%.6g final_alpha=%.6g warmup_epochs=%d "
            "decay_start_epoch=%d schedule=%s",
            args.local_jepa_alpha,
            args.local_jepa_final_alpha,
            args.local_jepa_warmup_epochs,
            args.local_jepa_decay_start_epoch,
            not args.no_local_jepa_alpha_schedule,
        )

    output_dir = Path(args.output_dir)
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
            logger.info("Unfroze CSPDarkNet feature extractor at epoch %d", epoch)

        local_jepa_alpha = 0.0
        if args.use_local_jepa:
            if args.no_local_jepa_alpha_schedule:
                local_jepa_alpha = args.local_jepa_alpha
            else:
                local_jepa_alpha = local_jepa_alpha_for_epoch(
                    epoch=epoch,
                    total_epochs=args.epochs,
                    peak_alpha=args.local_jepa_alpha,
                    final_alpha=args.local_jepa_final_alpha,
                    warmup_epochs=args.local_jepa_warmup_epochs,
                    decay_start_epoch=args.local_jepa_decay_start_epoch,
                )
            criterion.local_jepa_gain = local_jepa_alpha

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
            max_det=args.max_det,
        )
        scheduler.step()

        current_lrs = scheduler.get_last_lr()
        metrics = {
            **train_metrics,
            **val_metrics,
            "lr_backbone": current_lrs[0] if current_lrs else args.backbone_lr,
            "lr": current_lrs[-1] if current_lrs else args.lr,
            "local_jepa_alpha": local_jepa_alpha,
        }
        logger.info(
            "epoch %03d/%03d | loss=%.4f box=%.4f cls=%.4f dfl=%.4f jepa=%.4f alpha_jepa=%.4f | "
            "val_loss=%.4f mAP50=%.4f P@eval=%.4f R@eval=%.4f "
            "P@0.25=%.4f R@0.25=%.4f P@0.50=%.4f R@0.50=%.4f preds@eval=%d | "
            "lr_backbone=%.6g lr_head=%.6g",
            epoch,
            args.epochs,
            metrics["loss"],
            metrics["box"],
            metrics["cls"],
            metrics["dfl"],
            metrics["jepa"],
            metrics["local_jepa_alpha"],
            metrics["val_loss"],
            metrics["mAP50"],
            metrics["precision"],
            metrics["recall"],
            metrics["precision@0.25"],
            metrics["recall@0.25"],
            metrics["precision@0.5"],
            metrics["recall@0.5"],
            metrics["num_predictions"],
            metrics["lr_backbone"],
            metrics["lr"],
        )
        save_checkpoint(output_dir / "last.pth", model, epoch, classes, metrics)
        if metrics["mAP50"] > best_map + args.min_delta:
            best_map = metrics["mAP50"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.pth", model, epoch, classes, metrics)
            logger.info("Saved new best checkpoint: mAP50=%.6f at epoch %d", best_map, epoch)
        else:
            epochs_without_improvement += 1
            logger.info(
                "No mAP50 improvement for %d/%d epoch(s). Best mAP50=%.6f at epoch %d",
                epochs_without_improvement,
                args.early_stop_patience,
                best_map,
                best_epoch,
            )
            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                logger.info(
                    "Early stopping at epoch %d. Best mAP50=%.6f at epoch %d",
                    epoch,
                    best_map,
                    best_epoch,
                )
                break

    logger.info("Training finished. Best mAP50: %.6f", best_map)


if __name__ == "__main__":
    main()
