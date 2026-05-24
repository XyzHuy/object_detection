from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

from dataloader import build_dataloader
from inference import collect_images, load_annotation_index, load_model, predict_image
from loss import YOLOv8Loss
from trainer import evaluate_model, load_classes


def setup_eval_logger(log_dir: str | Path, checkpoint: str | Path) -> tuple[logging.Logger, Path]:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = Path(checkpoint).stem
    log_path = log_dir / f"eval_{checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("evaluate")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate a trained YOLOv8 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="final_public/public")
    parser.add_argument("--split", default="val")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--conf_threshold", type=float, default=0.001)
    parser.add_argument("--nms_iou", type=float, default=0.65)
    parser.add_argument("--max_det", type=int, default=100)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--predictions_output")
    parser.add_argument("--metrics_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger, log_path = setup_eval_logger(args.log_dir, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting evaluation")
    logger.info("Log file: %s", log_path)
    logger.info("Args: %s", json.dumps(vars(args), ensure_ascii=False, sort_keys=True))
    logger.info("Device: %s", device)

    classes = load_classes(args.data_root, split=args.split)
    model, checkpoint = load_model(args.checkpoint, len(classes), device)
    if checkpoint.get("classes"):
        classes = checkpoint["classes"]
    logger.info("Classes (%d): %s", len(classes), ", ".join(classes))
    if checkpoint.get("epoch") is not None:
        logger.info("Checkpoint epoch: %s", checkpoint["epoch"])
    if checkpoint.get("metrics"):
        logger.info("Checkpoint stored metrics: %s", json.dumps(checkpoint["metrics"], ensure_ascii=False, sort_keys=True))

    loader = build_dataloader(
        args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transforms=None,
        img_size=args.img_size,
    )
    criterion = YOLOv8Loss(num_classes=len(classes), strides=model.head.strides, reg_max=model.head.reg_max)
    metrics = evaluate_model(
        model,
        loader,
        criterion,
        device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.nms_iou,
        max_det=args.max_det,
    )

    metrics_text = json.dumps(metrics, ensure_ascii=False, indent=2)
    logger.info(
        "Evaluation metrics: mAP50=%.6f val_loss=%.6f P@eval=%.6f R@eval=%.6f "
        "P@0.25=%.6f R@0.25=%.6f P@0.50=%.6f R@0.50=%.6f num_predictions=%d",
        metrics["mAP50"],
        metrics["val_loss"],
        metrics["precision"],
        metrics["recall"],
        metrics["precision@0.25"],
        metrics["recall@0.25"],
        metrics["precision@0.5"],
        metrics["recall@0.5"],
        metrics["num_predictions"],
    )
    logger.info("Full metrics JSON:\n%s", metrics_text)
    if args.metrics_output:
        Path(args.metrics_output).write_text(metrics_text + "\n", encoding="utf-8")
        logger.info("Wrote metrics JSON to %s", args.metrics_output)

    if args.predictions_output:
        source = Path(args.data_root) / args.split / "images"
        annotation_index = load_annotation_index(args.data_root, args.split)
        predictions = []
        for image_path in collect_images(source):
            result = predict_image(
                model,
                image_path,
                classes,
                args.img_size,
                device,
                args.conf_threshold,
                args.nms_iou,
                args.max_det,
            )
            result["image_id"] = annotation_index.get(image_path.name, {}).get("id", image_path.name)
            predictions.append(result)
        Path(args.predictions_output).write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote predictions JSON to %s", args.predictions_output)


if __name__ == "__main__":
    main()
