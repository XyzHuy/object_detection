from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as trans
import torchvision.transforms.functional as trans_func
from PIL import Image

from metrics import non_max_suppression
from model import YOLOv8Scratch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox_image(image: Image.Image, img_size: int):
    width, height = image.size
    scale = img_size / max(width, height)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    pad_left = (img_size - new_w) // 2
    pad_top = (img_size - new_h) // 2
    canvas = Image.new("RGB", (img_size, img_size), (114, 114, 114))
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, {"scale": scale, "pad_left": pad_left, "pad_top": pad_top, "orig_size": (width, height)}


def preprocess(image: Image.Image, img_size: int, device: torch.device):
    image, meta = letterbox_image(image.convert("RGB"), img_size)
    tensor = trans_func.to_tensor(image)
    tensor = trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
    return tensor.unsqueeze(0).to(device), meta


def restore_boxes(boxes: torch.Tensor, meta: dict) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta["pad_left"]) / meta["scale"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta["pad_top"]) / meta["scale"]
    width, height = meta["orig_size"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    return boxes


def load_model(checkpoint_path: str | Path, num_classes: int, device: torch.device):
    model = YOLOv8Scratch(num_classes=num_classes, pretrained_backbone=False, use_cspdarknet=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device).eval()
    return model, checkpoint if isinstance(checkpoint, dict) else {}


def collect_images(source: str | Path) -> list[Path]:
    source = Path(source)
    if source.is_file():
        return [source]
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTS)


def load_annotation_index(data_root: str | Path, split: str) -> dict[str, dict]:
    path = Path(data_root) / "annotations" / f"{split}.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {Path(item["file_name"]).name: item for item in data["images"]}


@torch.no_grad()
def predict_image(
    model,
    image_path: Path,
    classes: list[str],
    img_size: int,
    device: torch.device,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    tensor, meta = preprocess(image, img_size, device)
    pred, _ = model(tensor)
    detections = non_max_suppression(pred, conf_threshold, iou_threshold, max_det=max_det)[0]
    boxes = restore_boxes(detections["boxes"].cpu(), meta)

    output_boxes = []
    for box, score, label in zip(boxes, detections["scores"].cpu(), detections["labels"].cpu()):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        if x2 <= x1 or y2 <= y1:
            continue
        output_boxes.append(
            {
                "class": classes[int(label)],
                "confidence": float(score),
                "bbox": [x1, y1, x2, y2],
            }
        )
    return {"image_id": image_path.name, "boxes": output_boxes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run YOLOv8 inference and write public predictions JSON")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", default="final_public/public/val/images")
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--data_root", default="final_public/public")
    parser.add_argument("--split", default="val")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--conf_threshold", type=float, default=0.001)
    parser.add_argument("--nms_iou", type=float, default=0.65)
    parser.add_argument("--max_det", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint.get("classes") if isinstance(checkpoint, dict) else None
    if classes is None:
        with (Path(args.data_root) / "annotations" / f"{args.split}.json").open("r", encoding="utf-8") as file:
            classes = json.load(file)["classes"]

    model, _ = load_model(args.checkpoint, len(classes), device)
    annotation_index = load_annotation_index(args.data_root, args.split)
    predictions = []

    for image_path in collect_images(args.source):
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
