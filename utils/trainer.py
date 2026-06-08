from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils.dataloader import albumentations_transform, build_dataloader
from utils.loss import YOLOv8Loss
from utils.metrics import detection_metrics, filter_predictions_by_score, non_max_suppression
from utils.model import YOLOv8Scratch


DEFAULT_IMG_SIZE = 768
DEFAULT_EPOCHS = 35
DEFAULT_SCHEDULER_EPOCHS = 80
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 0.00015
DEFAULT_BACKBONE_LR = 0.000025
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_FREEZE_BACKBONE_EPOCHS = 2
DEFAULT_EARLY_STOP_PATIENCE = 7
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_CONF_THRESHOLD = 0.001
DEFAULT_NMS_IOU = 0.65
DEFAULT_MAX_DET = 100
DEFAULT_SEED = 42
DEFAULT_NUM_RUNS = 1
DEFAULT_EMA_DECAY = 0.9999
DEFAULT_MOSAIC_OVERSAMPLE = 0.95
DEFAULT_BOX_EQUALIZER_OVERSAMPLE = "chair=1.0,cat=0.70,dog=0.60,car=0.50,person=0.35"
DEFAULT_NECK_DEPTH = 2
DEFAULT_HEAD_DEPTH = 3
DEFAULT_CLASS_WEIGHT_OVERRIDES = "chair=1.6"
DEFAULT_CLASS_TOPK_OVERRIDES = "chair=6"
DEFAULT_QUALITY_TARGET_FLOOR = 0.05


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


def null_logger() -> logging.Logger:
    logger = logging.getLogger(f"trainer.rank{os.environ.get('RANK', 'local')}")
    logger.setLevel(logging.CRITICAL)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def setup_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return True, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def set_sampler_epoch(loader, epoch: int) -> None:
    dataset = getattr(loader, "dataset", None)
    sampler = getattr(loader, "sampler", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def per_device_batch_size(global_batch_size: int, world_size: int) -> int:
    if world_size <= 1:
        return global_batch_size
    if global_batch_size < world_size:
        raise ValueError("--batch_size phải >= WORLD_SIZE khi dùng DDP")
    if global_batch_size % world_size != 0:
        raise ValueError("--batch_size phải chia hết cho WORLD_SIZE khi dùng DDP")
    return global_batch_size // world_size


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
    model = unwrap_model(model)
    backbone = getattr(model, "backbone", None)
    if hasattr(backbone, "set_feature_extractor_trainable"):
        backbone.set_feature_extractor_trainable(trainable)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def parse_class_weight_overrides(text: str | None) -> dict[str, float]:
    if not text:
        return {}
    overrides = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Mỗi mục --class_weight_overrides phải có dạng class=multiplier")
        class_name, value = item.split("=", 1)
        multiplier = float(value)
        if multiplier <= 0:
            raise ValueError("Multiplier trong --class_weight_overrides phải > 0")
        overrides[class_name.strip()] = multiplier
    return overrides


def parse_class_float_overrides(text: str | None, flag_name: str) -> dict[str, float]:
    if not text:
        return {}
    overrides = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Mỗi mục {flag_name} phải có dạng class=value")
        class_name, value = item.split("=", 1)
        number = float(value)
        if number < 0:
            raise ValueError(f"Giá trị trong {flag_name} phải >= 0")
        overrides[class_name.strip()] = number
    return overrides


def validate_class_names(values: dict, classes: list[str], flag_name: str) -> None:
    known = set(classes)
    unknown = sorted(class_name for class_name in values if class_name not in known)
    if unknown:
        raise ValueError(f"Class không tồn tại trong {flag_name}: {', '.join(unknown)}")


def parse_class_topk_overrides(text: str | None) -> dict[str, int]:
    if not text:
        return {}
    overrides = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Mỗi mục --class_topk_overrides phải có dạng class=top_k")
        class_name, value = item.split("=", 1)
        topk = int(value)
        if topk <= 0:
            raise ValueError("top_k trong --class_topk_overrides phải > 0")
        overrides[class_name.strip()] = topk
    return overrides


def build_class_topk(
    classes: list[str],
    default_topk: int,
    overrides: dict[str, int],
) -> torch.Tensor:
    if default_topk <= 0:
        raise ValueError("--assign_topk phải > 0")
    topk_by_class = torch.full((len(classes),), int(default_topk), dtype=torch.long)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    for class_name, topk in overrides.items():
        if class_name not in class_to_idx:
            raise ValueError(f"Class không tồn tại trong --class_topk_overrides: {class_name}")
        topk_by_class[class_to_idx[class_name]] = int(topk)
    return topk_by_class


def apply_class_weight_overrides(
    class_weights: torch.Tensor,
    classes: list[str],
    overrides: dict[str, float],
    renormalize: bool,
) -> torch.Tensor:
    if not overrides:
        return class_weights
    weights = class_weights.clone()
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    for class_name, multiplier in overrides.items():
        if class_name not in class_to_idx:
            raise ValueError(f"Class không tồn tại trong --class_weight_overrides: {class_name}")
        weights[class_to_idx[class_name]] *= multiplier
    if renormalize:
        weights = weights / weights.mean().clamp(min=1e-12)
    return weights


class ModelEMA:
    """EMA trọng số để validate và lưu checkpoint."""

    def __init__(self, model: torch.nn.Module, decay: float = DEFAULT_EMA_DECAY) -> None:
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def _decay(self) -> float:
        return self.decay * (1.0 - math.exp(-self.updates / 2000.0))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        decay = self._decay()
        model_state = model.state_dict()
        for name, ema_value in self.ema.state_dict().items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    amp: bool,
    epoch: int,
    ema: ModelEMA | None = None,
    distributed: bool = False,
    rank: int = 0,
) -> dict:
    model.train()
    running = {"loss": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0}
    seen_samples = 0
    progress = tqdm(loader, desc=f"train {epoch}", leave=False, disable=distributed and rank != 0)

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
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None and (not amp or scaler.get_scale() >= scale_before_step):
            ema.update(unwrap_model(model))

        batch_size = images.shape[0]
        seen_samples += batch_size
        running["loss"] += float(loss_items.loss.detach()) * batch_size
        running["box"] += float(loss_items.box_loss) * batch_size
        running["cls"] += float(loss_items.cls_loss) * batch_size
        running["dfl"] += float(loss_items.dfl_loss) * batch_size
        if not distributed or rank == 0:
            progress.set_postfix({key: value / max(seen_samples, 1) for key, value in running.items()})

    totals = torch.tensor(
        [running["loss"], running["box"], running["cls"], running["dfl"], float(seen_samples)],
        device=device,
        dtype=torch.float64,
    )
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    denom = max(float(totals[-1].item()), 1.0)
    return {
        "loss": float(totals[0].item() / denom),
        "box": float(totals[1].item() / denom),
        "cls": float(totals[2].item() / denom),
        "dfl": float(totals[3].item() / denom),
    }


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


def save_checkpoint(
    path: Path,
    model,
    epoch: int,
    classes: list[str],
    metrics: dict,
    model_config: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": state_dict_for_save(model, half=True),
            "classes": classes,
            "metrics": metrics,
            "model_config": model_config or {},
        },
        path,
    )


def state_dict_for_save(model, half: bool = False) -> dict:
    state = {}
    for key, value in model.state_dict().items():
        value = value.detach().cpu()
        if half and torch.is_floating_point(value):
            value = value.half()
        state[key] = value
    return state


def save_training_checkpoint(
    path: Path,
    model,
    eval_model,
    optimizer,
    scheduler,
    scaler,
    ema: ModelEMA | None,
    epoch: int,
    classes: list[str],
    metrics: dict,
    model_config: dict,
    best_map: float,
    best_epoch: int,
    epochs_without_improvement: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": state_dict_for_save(eval_model, half=True),
        "train_model": state_dict_for_save(unwrap_model(model), half=False),
        "classes": classes,
        "metrics": metrics,
        "model_config": model_config,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": state_dict_for_save(ema.ema, half=False) if ema is not None else None,
        "ema_updates": ema.updates if ema is not None else 0,
        "training_state": {
            "best_map": best_map,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        },
    }
    torch.save(checkpoint, path)


def load_resume_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    scaler,
    ema: ModelEMA | None,
    device: torch.device,
    logger: logging.Logger,
    classes: list[str],
) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint resume: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        logger.info("Đã resume trọng số từ state dict thô: %s", checkpoint_path)
        return {
            "start_epoch": 1,
            "best_map": -1.0,
            "best_epoch": 0,
            "epochs_without_improvement": 0,
            "full_state": False,
        }

    checkpoint_classes = checkpoint.get("classes")
    if checkpoint_classes is not None and list(checkpoint_classes) != list(classes):
        raise ValueError(
            "Class trong checkpoint resume không khớp class train hiện tại: "
            f"{checkpoint_classes} != {classes}"
        )

    state = checkpoint.get("train_model") or checkpoint.get("model")
    if state is None:
        raise KeyError(f"Checkpoint không có state 'train_model' hoặc 'model': {checkpoint_path}")
    model.load_state_dict(state)

    full_state = "train_model" in checkpoint and "optimizer" in checkpoint
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if ema is not None:
        if checkpoint.get("ema") is not None:
            ema.ema.load_state_dict(checkpoint["ema"])
            ema.updates = int(checkpoint.get("ema_updates", 0))
        else:
            ema.ema.load_state_dict(model.state_dict())

    epoch = int(checkpoint.get("epoch", 0))
    metrics = checkpoint.get("metrics") or {}
    training_state = checkpoint.get("training_state") or {}
    best_map = float(training_state.get("best_map", metrics.get("mAP50", -1.0)))
    best_epoch = int(training_state.get("best_epoch", epoch if best_map >= 0 else 0))
    epochs_without_improvement = int(training_state.get("epochs_without_improvement", 0))
    logger.info(
        "Đã resume checkpoint %s từ %s tại epoch %d (full_state=%s best_mAP50=%.6f best_epoch=%d)",
        "training" if full_state else "chỉ trọng số",
        checkpoint_path,
        epoch,
        full_state,
        best_map,
        best_epoch,
    )
    return {
        "start_epoch": epoch + 1 if epoch > 0 else 1,
        "best_map": best_map,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "full_state": full_state,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Huấn luyện YOLOv8 với backbone ConvNeXt V2 Tiny pretrained ImageNet")
    parser.add_argument("--data_root", default="final_public/public")
    parser.add_argument("--train_data", help="Alias CLI cho ./public/annotations/train.json")
    parser.add_argument("--val_data", help="Alias CLI cho ./public/annotations/val.json")
    parser.add_argument("--image_dir", help=" ảnh lấy theo data_root.")
    parser.add_argument("--val_image_dir", help=" ảnh val lấy theo data_root.")
    parser.add_argument("--checkpoint_dir", help="Alias cho --output_dir; lưu best.pth và last.pth trực tiếp tại đây.")
    parser.add_argument(
        "--resume",
        help="Resume train từ checkpoint. Phải dùng last.pth để giữ optimizer/scheduler/EMA.",
    )
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--scheduler_epochs",
        type=int,
        default=DEFAULT_SCHEDULER_EPOCHS,
        help="T_max của cosine LR theo epoch.",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--backbone_lr", type=float, default=DEFAULT_BACKBONE_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--log_dir", default="utils/logs")
    parser.add_argument("--conf_threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--nms_iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--max_det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument(
        "--class_weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dùng trọng số class sqrt-inverse từ train split cho loss phân loại positive.",
    )
    parser.add_argument(
        "--class_weight_overrides",
        default=DEFAULT_CLASS_WEIGHT_OVERRIDES,
        help="Multiplier class-weight positive, cách nhau bằng dấu phẩy, ví dụ chair=1.6,car=1.1.",
    )
    parser.add_argument(
        "--class_weight_renorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Chuẩn hóa lại class weights về mean 1 sau khi áp multiplier.",
    )
    parser.add_argument(
        "--assign_topk",
        type=int,
        default=10,
        help="top_k mặc định của TaskAligned assignment cho mỗi class.",
    )
    parser.add_argument(
        "--class_topk_overrides",
        default=DEFAULT_CLASS_TOPK_OVERRIDES,
        help="Override top_k theo class, cách nhau bằng dấu phẩy, ví dụ chair=6,car=12.",
    )
    parser.add_argument(
        "--quality_targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale target phân loại positive theo quality thay vì dùng nhãn 1.0.",
    )
    parser.add_argument(
        "--quality_target_floor",
        type=float,
        default=DEFAULT_QUALITY_TARGET_FLOOR,
        help="Giá trị target positive tối thiểu khi bật --quality_targets.",
    )
    parser.add_argument("--freeze_backbone_epochs", type=int, default=DEFAULT_FREEZE_BACKBONE_EPOCHS)
    parser.add_argument("--early_stop_patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--min_delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--num_runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--EMA",
        "--ema",
        dest="ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dùng trọng số EMA để validate và lưu checkpoint.",
    )
    parser.add_argument("--ema_decay", type=float, default=DEFAULT_EMA_DECAY)
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument(
        "--mosaic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Thêm sample oversampling mosaic 4 ảnh kiểu YOLO vào epoch train.",
    )
    parser.add_argument(
        "--mosaic_oversample",
        type=float,
        default=DEFAULT_MOSAIC_OVERSAMPLE,
        help=(
            "Số sample mosaic thêm trên mỗi ảnh train gốc. Với equalizer mặc định, "
            "0.95 giữ mosaic khoảng 30%% số sample train."
        ),
    )
    parser.add_argument("--mosaic_p", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--box_type_equalizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Oversample sample train để bù bucket kích thước bbox thiếu theo class trước Albumentations.",
    )
    parser.add_argument(
        "--box_shape_equalizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Oversample sample train để bù bucket dáng bbox thiếu theo class trước Albumentations.",
    )
    parser.add_argument(
        "--box_equalizer_oversample",
        default=DEFAULT_BOX_EQUALIZER_OVERSAMPLE,
        help=(
            "Mức oversample equalizer theo class, cách nhau bằng dấu phẩy, ví dụ "
            "chair=1.0,cat=0.5. 1.0 thêm một sample giả cho mỗi thiếu hụt bucket."
        ),
    )
    parser.add_argument(
        "--neck_depth",
        type=int,
        default=DEFAULT_NECK_DEPTH,
        help="Độ sâu block C2f trong PAN/FPN neck. 1 giống baseline cũ; 2 mạnh hơn.",
    )
    parser.add_argument(
        "--head_depth",
        type=int,
        default=DEFAULT_HEAD_DEPTH,
        help="Số lớp Conv trước mỗi detect output. 2 giống baseline cũ; 3 sâu hơn nhẹ.",
    )
    parser.add_argument("--scratch_backbone", action="store_true")
    parser.add_argument(
        "--p2_head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Thêm detect head P2 stride-4 cho object nhỏ.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Đặt -1 để tắt seed cố định.")
    parser.add_argument("--local_rank", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    args.flat_checkpoint_dir = False
    if args.train_data:
        train_data = Path(args.train_data)
        if train_data.name != "train.json":
            raise ValueError("--train_data nên trỏ tới annotations/train.json")
        args.data_root = str(train_data.parent.parent)
    if args.checkpoint_dir:
        args.output_dir = args.checkpoint_dir
        args.flat_checkpoint_dir = True
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema_decay phải nằm trong [0, 1)")
    if args.mosaic_p is not None:
        args.mosaic_oversample = args.mosaic_p
    delattr(args, "mosaic_p")
    if args.mosaic_oversample < 0:
        raise ValueError("--mosaic_oversample phải >= 0")
    if args.scheduler_epochs <= 0:
        raise ValueError("--scheduler_epochs phải > 0")
    if args.neck_depth < 1:
        raise ValueError("--neck_depth phải >= 1")
    if args.head_depth < 1:
        raise ValueError("--head_depth phải >= 1")
    if args.assign_topk <= 0:
        raise ValueError("--assign_topk phải > 0")
    if not 0.0 <= args.quality_target_floor <= 1.0:
        raise ValueError("--quality_target_floor phải nằm trong [0, 1]")
    args.class_weight_overrides = parse_class_weight_overrides(args.class_weight_overrides)
    args.class_topk_overrides = parse_class_topk_overrides(args.class_topk_overrides)
    args.box_equalizer_oversample = parse_class_float_overrides(
        args.box_equalizer_oversample,
        "--box_equalizer_oversample",
    )
    return args


def train_single_run(
    args: argparse.Namespace,
    run_idx: int,
    output_dir: Path,
    log_dir: Path,
    distributed: bool = False,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
) -> None:
    if is_main_process(rank):
        logger, log_path = setup_logger(log_dir)
    else:
        logger, log_path = null_logger(), None
    if distributed and torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(args.data_root)
    validate_class_names(args.box_equalizer_oversample, classes, "--box_equalizer_oversample")
    run_seed = args.seed + run_idx if args.seed >= 0 else None
    local_batch_size = per_device_batch_size(args.batch_size, world_size)

    logger.info("Bắt đầu lượt train %d/%d", run_idx + 1, args.num_runs)
    logger.info("File log: %s", log_path)
    logger.info("Thư mục output: %s", output_dir)
    logger.info("Tham số: %s", json.dumps(vars(args), ensure_ascii=False, sort_keys=True))
    logger.info("Thiết bị: %s", device)
    logger.info("Distributed: bật=%s rank=%d local_rank=%d world_size=%d", distributed, rank, local_rank, world_size)
    logger.info(
        "Batch size: global=%d mỗi_thiết_bị=%d",
        args.batch_size,
        local_batch_size,
    )
    logger.info("Classes (%d): %s", len(classes), ", ".join(classes))
    logger.info("Chỉ số run: %d", run_idx)

    if run_seed is not None:
        set_seed(run_seed)
        logger.info("Seed: %d", run_seed)
    else:
        logger.info("Seed: tắt")
    transforms = None if args.no_aug else albumentations_transform()
    train_loader = build_dataloader(
        args.data_root,
        split="train",
        batch_size=local_batch_size,
        num_workers=args.num_workers,
        transforms=transforms,
        img_size=args.img_size,
        seed=run_seed,
        box_type_equalizer=args.box_type_equalizer,
        box_shape_equalizer=args.box_shape_equalizer,
        box_equalizer_oversample=args.box_equalizer_oversample,
        mosaic=args.mosaic and not args.no_aug,
        mosaic_oversample=args.mosaic_oversample,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    if is_main_process(rank):
        sample_plan_stats = getattr(train_loader.dataset, "sample_plan_stats", None)
        if sample_plan_stats is not None:
            logger.info(
                "Kế hoạch sample train: %s",
                json.dumps(sample_plan_stats, ensure_ascii=False, sort_keys=True),
            )
    if is_main_process(rank) and args.box_type_equalizer:
        equalizer = getattr(train_loader.dataset, "box_type_equalizer", None)
        if equalizer is not None:
            logger.info(
                "Thống kê equalizer kích thước bbox: %s",
                json.dumps(equalizer.stats(), ensure_ascii=False, sort_keys=True),
            )
    if is_main_process(rank) and args.box_shape_equalizer:
        equalizer = getattr(train_loader.dataset, "box_shape_equalizer", None)
        if equalizer is not None:
            logger.info(
                "Thống kê equalizer dáng bbox: %s",
                json.dumps(equalizer.stats(), ensure_ascii=False, sort_keys=True),
            )
    val_loader = None
    if is_main_process(rank):
        val_loader = build_dataloader(
            args.data_root,
            split="val",
            batch_size=local_batch_size,
            num_workers=args.num_workers,
            transforms=None,
            img_size=args.img_size,
            seed=run_seed + 1 if run_seed is not None else None,
        )

    model = YOLOv8Scratch(
        num_classes=len(classes),
        pretrained_backbone=not args.scratch_backbone,
        use_pretrained_backbone=not args.scratch_backbone,
        use_p2=args.p2_head,
        neck_depth=args.neck_depth,
        head_depth=args.head_depth,
    ).to(device)
    model_config = {
        "use_p2": args.p2_head,
        "use_pretrained_backbone": not args.scratch_backbone,
        "backbone_name": "convnextv2_tiny",
        "neck_depth": args.neck_depth,
        "head_depth": args.head_depth,
    }
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None
    class_weights, class_counts = compute_class_weights(
        data_root=args.data_root,
        classes=classes,
        split="train",
    )
    if not args.class_weighting:
        class_weights = torch.ones_like(class_weights)
    base_class_weights = class_weights.clone()
    class_weights = apply_class_weight_overrides(
        class_weights=class_weights,
        classes=classes,
        overrides=args.class_weight_overrides,
        renormalize=args.class_weight_renorm,
    )
    class_topk = build_class_topk(
        classes=classes,
        default_topk=args.assign_topk,
        overrides=args.class_topk_overrides,
    )
    logger.info("Số bbox theo class train: %s", json.dumps(class_counts, ensure_ascii=False, sort_keys=True))
    logger.info(
        "Class weighting: bật=%s scheme=sqrt_inverse multipliers=%s renorm=%s base_weights=%s weights=%s",
        args.class_weighting,
        json.dumps(args.class_weight_overrides, ensure_ascii=False, sort_keys=True),
        args.class_weight_renorm,
        json.dumps({class_name: round(float(weight), 6) for class_name, weight in zip(classes, base_class_weights)}),
        json.dumps({class_name: round(float(weight), 6) for class_name, weight in zip(classes, class_weights)}),
    )
    logger.info(
        "Assignment top_k: mặc_định=%d overrides=%s theo_class=%s",
        args.assign_topk,
        json.dumps(args.class_topk_overrides, ensure_ascii=False, sort_keys=True),
        json.dumps({class_name: int(topk) for class_name, topk in zip(classes, class_topk)}),
    )
    logger.info(
        "Oversampling mosaic: bật=%s ratio=%.3f",
        args.mosaic and not args.no_aug,
        args.mosaic_oversample,
    )
    criterion = YOLOv8Loss(
        num_classes=len(classes),
        strides=model.head.strides,
        reg_max=model.head.reg_max,
        topk=args.assign_topk,
        topk_by_class=class_topk,
        class_weights=class_weights,
        quality_targets=args.quality_targets,
        quality_target_floor=args.quality_target_floor,
    )
    logger.info(
        "Quality targets: bật=%s floor=%.6g",
        args.quality_targets,
        args.quality_target_floor,
    )

    optimizer = build_optimizer(
        model,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.scheduler_epochs, 1),
        eta_min=min(args.lr, args.backbone_lr) * 0.05,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))
    start_epoch = 1
    best_map = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    if args.resume:
        resume_state = load_resume_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            device=device,
            logger=logger,
            classes=classes,
        )
        start_epoch = int(resume_state["start_epoch"])
        best_map = float(resume_state["best_map"])
        best_epoch = int(resume_state["best_epoch"])
        epochs_without_improvement = int(resume_state["epochs_without_improvement"])
    if distributed:
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        else:
            model = DDP(model, find_unused_parameters=True)
    set_backbone_trainable(model, args.freeze_backbone_epochs <= 0 or start_epoch > args.freeze_backbone_epochs)
    logger.info(
        "Nhóm optimizer: lr=%.6g backbone_lr=%.6g weight_decay=%.6g",
        args.lr,
        args.backbone_lr,
        args.weight_decay,
    )
    logger.info(
        "Scheduler: cosine_T_max_epochs=%d train_epochs=%d",
        args.scheduler_epochs,
        args.epochs,
    )
    logger.info("EMA: bật=%s decay=%.6g", args.ema, args.ema_decay)
    logger.info("Cấu hình model: %s", json.dumps(model_config, ensure_ascii=False, sort_keys=True))

    if start_epoch > args.epochs:
        logger.info("Epoch resume %d vượt quá --epochs=%d; không chạy epoch train nào.", start_epoch, args.epochs)

    for epoch in range(start_epoch, args.epochs + 1):
        set_sampler_epoch(train_loader, epoch)
        if is_main_process(rank):
            sample_plan_stats = getattr(train_loader.dataset, "sample_plan_stats", None)
            if sample_plan_stats is not None:
                logger.info(
                    "Kế hoạch sample train epoch %d đã làm mới: items=%d by_mode=%s mode_ratios=%s",
                    epoch,
                    sample_plan_stats["dataset_items"],
                    json.dumps(sample_plan_stats["by_mode"], ensure_ascii=False, sort_keys=True),
                    json.dumps(sample_plan_stats["mode_ratios"], ensure_ascii=False, sort_keys=True),
                )
        if epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
            logger.info("Đã mở đóng băng feature extractor ConvNeXt V2 Tiny tại epoch %d", epoch)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            device.type == "cuda" and not args.no_amp,
            epoch,
            ema=ema,
            distributed=distributed,
            rank=rank,
        )
        scheduler.step()
        stop_training = False

        if is_main_process(rank):
            eval_model = ema.ema if ema is not None else unwrap_model(model)
            val_metrics = evaluate_model(
                eval_model,
                val_loader,
                criterion,
                device,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.nms_iou,
                max_det=args.max_det,
            )
            current_lrs = scheduler.get_last_lr()
            metrics = {
                **train_metrics,
                **val_metrics,
                "lr_backbone": current_lrs[0] if current_lrs else args.backbone_lr,
                "lr": current_lrs[-1] if current_lrs else args.lr,
                "run": run_idx,
                "seed": run_seed,
                "experiment": experiment_name(),
                "ema": ema is not None,
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
            if metrics["mAP50"] > best_map + args.min_delta:
                best_map = metrics["mAP50"]
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(output_dir / "best.pth", eval_model, epoch, classes, metrics, model_config)
                logger.info("Đã lưu checkpoint tốt nhất mới: mAP50=%.6f tại epoch %d", best_map, epoch)
            else:
                epochs_without_improvement += 1
                logger.info(
                    "mAP50 chưa cải thiện trong %d/%d epoch. mAP50 tốt nhất=%.6f tại epoch %d",
                    epochs_without_improvement,
                    args.early_stop_patience,
                    best_map,
                    best_epoch,
                )
                if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                    logger.info(
                        "Dừng sớm tại epoch %d. mAP50 tốt nhất=%.6f tại epoch %d",
                        epoch,
                        best_map,
                        best_epoch,
                    )
                    stop_training = True
            save_training_checkpoint(
                output_dir / "last.pth",
                model=model,
                eval_model=eval_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                epoch=epoch,
                classes=classes,
                metrics=metrics,
                model_config=model_config,
                best_map=best_map,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
            )
            logger.info("Đã lưu checkpoint resume: %s", output_dir / "last.pth")

        if distributed:
            stop_tensor = torch.tensor(1 if stop_training else 0, device=device, dtype=torch.int)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())
        if stop_training:
            break

    if is_main_process(rank):
        logger.info("Train hoàn tất. mAP50 tốt nhất: %.6f tại epoch %d", best_map, best_epoch)


def main() -> None:
    args = parse_args()
    if args.num_runs < 1:
        raise ValueError("--num_runs phải >= 1")

    distributed, rank, local_rank, world_size = setup_distributed()
    try:
        experiment = experiment_name()
        for run_idx in range(args.num_runs):
            output_dir = Path(args.output_dir)
            log_dir = Path(args.log_dir)
            if not getattr(args, "flat_checkpoint_dir", False) or args.num_runs > 1:
                output_dir = run_dir(args.output_dir, experiment, run_idx)
                log_dir = run_dir(args.log_dir, experiment, run_idx)
            train_single_run(
                args=args,
                run_idx=run_idx,
                output_dir=output_dir,
                log_dir=log_dir,
                distributed=distributed,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
            )
            if distributed:
                dist.barrier()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()


# python3 train.py
# python3 train.py --num_runs 3
