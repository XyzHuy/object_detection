from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
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
DEFAULT_ALTER_BEST_MIN_MAP = 0.8
DEFAULT_CONF_THRESHOLD = 0.001
DEFAULT_NMS_IOU = 0.65
DEFAULT_MAX_DET = 100
DEFAULT_SEED = 42
DEFAULT_NUM_RUNS = 1


def experiment_name() -> str:
    return "base"


def run_dir(root: str | Path, experiment: str, run_idx: int) -> Path:
    return Path(root) / experiment / f"run_{run_idx}"


def setup_logger(log_dir: str | Path) -> tuple[logging.Logger, Path]:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
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


def compute_class_weights(
    data_root: str | Path,
    classes: list[str],
    split: str = "train",
) -> tuple[torch.Tensor, dict[str, int]]:
    annotation_path = Path(data_root) / "annotations" / f"{split}.json"
    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    counts_by_class = {class_name: 0 for class_name in classes}
    for ann in data["annotations"]:
        class_name = ann["class"]
        if class_name in counts_by_class:
            counts_by_class[class_name] += 1

    counts = torch.tensor([counts_by_class[class_name] for class_name in classes], dtype=torch.float32)
    safe_counts = counts.clamp(min=1.0)
    weights = 1.0 / safe_counts.sqrt()
    weights = weights / weights.mean().clamp(min=1e-12)
    return weights, counts_by_class


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


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


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
    max_det: int = DEFAULT_MAX_DET,
    report_thresholds=(0.05, 0.25, 0.50),
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
    parser.add_argument(
        "--class_weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sqrt-inverse class weights from the train split for positive classification loss.",
    )
    parser.add_argument("--freeze_backbone_epochs", type=int, default=DEFAULT_FREEZE_BACKBONE_EPOCHS)
    parser.add_argument("--early_stop_patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--min_delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument(
        "--alter_best_min_map",
        type=float,
        default=DEFAULT_ALTER_BEST_MIN_MAP,
        help="Save alter_best.pth as the latest epoch whose mAP50 is at least this value. Set <0 to disable.",
    )
    parser.add_argument("--num_runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--scratch_backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Set to -1 to disable fixed seeding.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic PyTorch kernels when available.")
    return parser.parse_args()


def train_single_run(args: argparse.Namespace, run_idx: int, output_dir: Path, log_dir: Path) -> None:
    logger, log_path = setup_logger(log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(args.data_root)
    run_seed = args.seed + run_idx if args.seed >= 0 else None

    logger.info("Starting training run %d/%d", run_idx + 1, args.num_runs)
    logger.info("Log file: %s", log_path)
    logger.info("Output dir: %s", output_dir)
    logger.info("Args: %s", json.dumps(vars(args), ensure_ascii=False, sort_keys=True))
    logger.info("Device: %s", device)
    logger.info("Classes (%d): %s", len(classes), ", ".join(classes))
    logger.info("Run index: %d", run_idx)

    if run_seed is not None:
        set_seed(run_seed, deterministic=args.deterministic)
        logger.info("Seed: %d deterministic=%s", run_seed, args.deterministic)
    else:
        logger.info("Seed: disabled")
    transforms = None if args.no_aug else albumentations_transform()
    train_loader = build_dataloader(
        args.data_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transforms=transforms,
        img_size=args.img_size,
        seed=run_seed,
    )
    val_loader = build_dataloader(
        args.data_root,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transforms=None,
        img_size=args.img_size,
        seed=run_seed + 1 if run_seed is not None else None,
    )

    model = YOLOv8Scratch(
        num_classes=len(classes),
        pretrained_backbone=not args.scratch_backbone,
        use_cspdarknet=not args.scratch_backbone,
    ).to(device)
    class_weights, class_counts = compute_class_weights(
        data_root=args.data_root,
        classes=classes,
        split="train",
    )
    if not args.class_weighting:
        class_weights = torch.ones_like(class_weights)
    logger.info("Train class counts: %s", json.dumps(class_counts, ensure_ascii=False, sort_keys=True))
    logger.info(
        "Class weighting: enabled=%s scheme=sqrt_inverse weights=%s",
        args.class_weighting,
        json.dumps({class_name: round(float(weight), 6) for class_name, weight in zip(classes, class_weights)}),
    )
    criterion = YOLOv8Loss(
        num_classes=len(classes),
        strides=model.head.strides,
        reg_max=model.head.reg_max,
        class_weights=class_weights,
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
    alter_best_epoch = 0
    alter_best_map = -1.0
    epochs_without_improvement = 0
    logger.info(
        "Optimizer groups: lr=%.6g backbone_lr=%.6g weight_decay=%.6g",
        args.lr,
        args.backbone_lr,
        args.weight_decay,
    )

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
            logger.info("Unfroze CSPDarkNet feature extractor at epoch %d", epoch)

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
            "run": run_idx,
            "seed": run_seed,
            "experiment": experiment_name(),
        }
        logger.info(
            "epoch %03d/%03d | loss=%.4f box=%.4f cls=%.4f dfl=%.4f | "
            "val_loss=%.4f mAP50=%.4f P@eval=%.4f R@eval=%.4f "
            "P@0.25=%.4f R@0.25=%.4f P@0.50=%.4f R@0.50=%.4f preds@eval=%d | "
            "lr_backbone=%.6g lr_head=%.6g",
            epoch,
            args.epochs,
            metrics["loss"],
            metrics["box"],
            metrics["cls"],
            metrics["dfl"],
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
        if args.alter_best_min_map >= 0 and metrics["mAP50"] >= args.alter_best_min_map:
            alter_best_epoch = epoch
            alter_best_map = metrics["mAP50"]
            save_checkpoint(output_dir / "alter_best.pth", model, epoch, classes, metrics)
            logger.info(
                "Saved alter_best checkpoint: mAP50=%.6f at epoch %d "
                "(latest epoch with mAP50 >= %.6f)",
                alter_best_map,
                alter_best_epoch,
                args.alter_best_min_map,
            )
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

    if alter_best_epoch:
        logger.info("Training finished. Best mAP50: %.6f | alter_best mAP50: %.6f at epoch %d", best_map, alter_best_map, alter_best_epoch)
    else:
        logger.info("Training finished. Best mAP50: %.6f | no alter_best checkpoint met mAP50 >= %.6f", best_map, args.alter_best_min_map)


def main() -> None:
    args = parse_args()
    if args.num_runs < 1:
        raise ValueError("--num_runs must be >= 1")

    experiment = experiment_name()
    for run_idx in range(args.num_runs):
        train_single_run(
            args=args,
            run_idx=run_idx,
            output_dir=run_dir(args.output_dir, experiment, run_idx),
            log_dir=run_dir(args.log_dir, experiment, run_idx),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


# python3 train.py
# python3 train.py --num_runs 3
