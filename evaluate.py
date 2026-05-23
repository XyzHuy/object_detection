from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataloader import build_dataloader
from inference import collect_images, load_annotation_index, load_model, predict_image
from loss import YOLOv8Loss
from trainer import evaluate_model, load_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate a trained YOLOv8 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="final_public/public")
    parser.add_argument("--split", default="val")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--conf_threshold", type=float, default=0.001)
    parser.add_argument("--nms_iou", type=float, default=0.65)
    parser.add_argument("--max_det", type=int, default=100)
    parser.add_argument("--predictions_output")
    parser.add_argument("--metrics_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(args.data_root, split=args.split)
    model, checkpoint = load_model(args.checkpoint, len(classes), device)
    if checkpoint.get("classes"):
        classes = checkpoint["classes"]

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
    )

    metrics_text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(metrics_text)
    if args.metrics_output:
        Path(args.metrics_output).write_text(metrics_text + "\n", encoding="utf-8")

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


if __name__ == "__main__":
    main()
