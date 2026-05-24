from __future__ import annotations

import argparse
import importlib.util
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
    parser.add_argument("--predictions_output", default = "predictions.json")
    parser.add_argument("--metrics_output", default = "metrics.json")
    parser.add_argument("--official_score_output", default ="val_score.json")
    parser.add_argument("--official_evaluator", default="final_public/public/tools/evaluate_predictions.py")
    parser.add_argument("--official_ground_truth")
    return parser.parse_args()


def run_official_evaluator(
    evaluator_path: str | Path,
    ground_truth_path: str | Path,
    predictions: list[dict],
    output_path: str | Path | None,
    max_detections_per_image: int,
) -> dict:
    evaluator_path = Path(evaluator_path)
    spec = importlib.util.spec_from_file_location("public_evaluate_predictions", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official evaluator from {evaluator_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ground_truth = module.load_json(Path(ground_truth_path))
    classes, image_info = module.validate_ground_truth(ground_truth)
    normalized = module.normalize_predictions(
        predictions,
        classes=classes,
        image_info=image_info,
        max_detections_per_image=max_detections_per_image,
        require_complete=True,
    )
    result = module.evaluate(
        ground_truth=ground_truth,
        predictions=normalized,
        classes=classes,
        iou_threshold=0.5,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result



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

    predictions = None
    if args.predictions_output or args.official_score_output:
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
        if args.predictions_output:
            Path(args.predictions_output).write_text(
                json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            logger.info("Wrote predictions JSON to %s", args.predictions_output)
        if args.official_score_output:
            official_ground_truth = args.official_ground_truth or str(Path(args.data_root) / "annotations" / f"{args.split}.json")
            official_metrics = run_official_evaluator(
                evaluator_path=args.official_evaluator,
                ground_truth_path=official_ground_truth,
                predictions=predictions or [],
                output_path=args.official_score_output,
                max_detections_per_image=args.max_det,
            )
            logger.info(
            "Official evaluator: mAP@0.5=%.6f micro_P=%.6f micro_R=%.6f predictions=%d",
            official_metrics["mAP@0.5"],
            official_metrics["micro_precision"],
            official_metrics["micro_recall"],
            official_metrics["num_predictions"],
        )
        logger.info("Wrote official score JSON to %s", args.official_score_output)

if __name__ == "__main__":
    main()

# python3 evaluate.py --checkpoint checkpoints/base/best.pth --log_dir logs/eval_base --predictions_output predictions_base.json --metrics_output metrics_base.json --official_score_output score_base.json
# python3 evaluate.py --checkpoint checkpoints/local_JEPA/best.pth --log_dir logs/eval_local_JEPA --predictions_output predictions_local_JEPA.json --metrics_output metrics_local_JEPA.json --official_score_output score_local_JEPA.json