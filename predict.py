from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from utils.inference import collect_images, load_class_thresholds, load_model, predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Chạy suy luận và ghi JSON dự đoán")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="models/best.pth")
    parser.add_argument(
        "--thresholds",
        help="File JSON ngưỡng đã tune. Mặc định dùng models/best_thresholds.json nếu có.",
    )
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--conf_threshold", type=float)
    parser.add_argument("--nms_iou", type=float)
    parser.add_argument("--max_det", type=int)
    parser.add_argument("--nms_max_det", type=int)
    return parser.parse_args()


def load_threshold_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_classes_from_checkpoint(checkpoint: dict, checkpoint_path: Path) -> list[str]:
    classes = checkpoint.get("classes") if isinstance(checkpoint, dict) else None
    if classes:
        return classes

    fallback = Path("public/annotations/train.json")
    if fallback.exists():
        return json.loads(fallback.read_text(encoding="utf-8"))["classes"]
    raise ValueError(f"Checkpoint {checkpoint_path} không lưu classes và không tìm thấy {fallback}")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    threshold_path = Path(args.thresholds) if args.thresholds else Path("models/best_thresholds.json")
    threshold_config = load_threshold_config(threshold_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = load_classes_from_checkpoint(checkpoint, checkpoint_path)
    model, _ = load_model(checkpoint_path, len(classes), device)

    class_thresholds = load_class_thresholds(threshold_path, classes) if threshold_config else None
    img_size = args.img_size or int(threshold_config.get("img_size", 768))
    conf_threshold = args.conf_threshold
    if conf_threshold is None:
        conf_threshold = float(threshold_config.get("base_conf_threshold", 0.001))
    nms_iou = args.nms_iou
    if nms_iou is None:
        nms_iou = float(threshold_config.get("nms_iou", 0.65))
    max_det = args.max_det or int(threshold_config.get("max_det", 100))
    nms_max_det = args.nms_max_det or int(threshold_config.get("nms_max_det", max_det))

    predictions = []
    for image_path in collect_images(args.image_dir):
        result = predict_image(
            model=model,
            image_path=image_path,
            classes=classes,
            img_size=img_size,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=nms_iou,
            max_det=nms_max_det,
            class_thresholds=class_thresholds,
        )
        result["boxes"] = sorted(result["boxes"], key=lambda item: item["confidence"], reverse=True)[:max_det]
        predictions.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã ghi dự đoán cho {len(predictions)} ảnh vào {output_path}")


if __name__ == "__main__":
    main()
""" 
python predict.py \
  --image_dir public/val/images \
  --output predictions.json """
