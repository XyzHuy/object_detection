#!/usr/bin/env python3
"""Validate object detection predictions and compute mAP@0.5."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate object detection predictions.")
    parser.add_argument("--ground_truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument(
        "--max_detections_per_image",
        type=int,
        default=100,
        help="Keep at most this many detections per image after sorting by confidence.",
    )
    parser.add_argument(
        "--allow_missing_images",
        action="store_true",
        help="Allow predictions.json to omit images. By default every image must appear once.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def validate_ground_truth(data: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if not isinstance(data, dict):
        raise ValueError("Ground truth must be a JSON object.")

    classes = data.get("classes")
    images = data.get("images")
    annotations = data.get("annotations")

    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ValueError("Ground truth field 'classes' must be a list of strings.")
    if not isinstance(images, list):
        raise ValueError("Ground truth field 'images' must be a list.")
    if not isinstance(annotations, list):
        raise ValueError("Ground truth field 'annotations' must be a list.")

    image_info: dict[str, dict[str, Any]] = {}
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("Each ground truth image entry must be an object.")
        image_id = image.get("id")
        width = image.get("width")
        height = image.get("height")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("Each ground truth image needs a non-empty string id.")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(f"Image {image_id} has invalid width/height.")
        image_info[image_id] = image

    class_set = set(classes)
    for ann in annotations:
        if not isinstance(ann, dict):
            raise ValueError("Each ground truth annotation must be an object.")
        image_id = ann.get("image_id")
        class_name = ann.get("class")
        bbox = ann.get("bbox")
        if image_id not in image_info:
            raise ValueError(f"Annotation references unknown image_id: {image_id}")
        if class_name not in class_set:
            raise ValueError(f"Annotation uses unknown class: {class_name}")
        validate_bbox(bbox, image_info[image_id], context=f"ground truth {image_id}")

    return classes, image_info


def validate_bbox(bbox: Any, image: dict[str, Any], context: str) -> list[float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox in {context}: expected [xmin, ymin, xmax, ymax].")
    if not all(isinstance(value, (int, float)) for value in bbox):
        raise ValueError(f"Invalid bbox in {context}: coordinates must be numeric.")

    xmin, ymin, xmax, ymax = [float(value) for value in bbox]
    width = float(image["width"])
    height = float(image["height"])

    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
        raise ValueError(f"Invalid bbox in {context}: coordinates outside image bounds.")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid bbox in {context}: xmax/ymax must be larger than xmin/ymin.")

    return [xmin, ymin, xmax, ymax]


def normalize_predictions(
    data: Any,
    classes: list[str],
    image_info: dict[str, dict[str, Any]],
    max_detections_per_image: int,
    require_complete: bool,
) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("Predictions must be a JSON array.")

    class_set = set(classes)
    seen_images: set[str] = set()
    normalized: list[dict[str, Any]] = []

    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("Each prediction entry must be an object.")

        image_id = entry.get("image_id")
        boxes = entry.get("boxes")
        if image_id not in image_info:
            raise ValueError(f"Prediction references unknown image_id: {image_id}")
        if image_id in seen_images:
            raise ValueError(f"Duplicate prediction entry for image_id: {image_id}")
        if not isinstance(boxes, list):
            raise ValueError(f"Prediction for {image_id} must contain a boxes list.")

        seen_images.add(image_id)
        image = image_info[image_id]

        image_boxes = []
        for index, box in enumerate(boxes):
            if not isinstance(box, dict):
                raise ValueError(f"Prediction box {index} for {image_id} must be an object.")
            class_name = box.get("class")
            confidence = box.get("confidence")
            bbox = box.get("bbox")
            if class_name not in class_set:
                raise ValueError(f"Prediction for {image_id} uses unknown class: {class_name}")
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                raise ValueError(f"Prediction for {image_id} has invalid confidence: {confidence}")

            image_boxes.append(
                {
                    "image_id": image_id,
                    "class": class_name,
                    "confidence": float(confidence),
                    "bbox": validate_bbox(bbox, image, context=f"prediction {image_id}"),
                }
            )

        image_boxes.sort(key=lambda item: item["confidence"], reverse=True)
        normalized.extend(image_boxes[:max_detections_per_image])

    if require_complete:
        missing = sorted(set(image_info) - seen_images)
        if missing:
            preview = ", ".join(missing[:10])
            suffix = "..." if len(missing) > 10 else ""
            raise ValueError(f"Predictions are missing {len(missing)} image(s): {preview}{suffix}")

    return normalized


def group_ground_truth(
    data: dict[str, Any], classes: list[str]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        class_name: defaultdict(list) for class_name in classes
    }

    for ann in data["annotations"]:
        grouped[ann["class"]][ann["image_id"]].append(
            {"bbox": [float(value) for value in ann["bbox"]], "matched": False}
        )

    return grouped


def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]

    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])

    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def evaluate(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    classes: list[str],
    iou_threshold: float,
) -> dict[str, Any]:
    gt_by_class = group_ground_truth(ground_truth, classes)
    pred_by_class: dict[str, list[dict[str, Any]]] = {class_name: [] for class_name in classes}
    for pred in predictions:
        pred_by_class[pred["class"]].append(pred)

    per_class = {}
    aps = []
    total_tp = 0
    total_fp = 0
    total_gt = 0

    for class_name in classes:
        class_gt = gt_by_class[class_name]
        num_gt = sum(len(items) for items in class_gt.values())
        class_preds = sorted(
            pred_by_class[class_name], key=lambda item: item["confidence"], reverse=True
        )

        tp_flags: list[int] = []
        fp_flags: list[int] = []

        for pred in class_preds:
            candidates = class_gt.get(pred["image_id"], [])
            best_iou = 0.0
            best_index = -1

            for index, gt in enumerate(candidates):
                if gt["matched"]:
                    continue
                iou = bbox_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index >= 0 and best_iou >= iou_threshold:
                candidates[best_index]["matched"] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        cumulative_tp = []
        cumulative_fp = []
        tp_sum = 0
        fp_sum = 0
        for tp, fp in zip(tp_flags, fp_flags):
            tp_sum += tp
            fp_sum += fp
            cumulative_tp.append(tp_sum)
            cumulative_fp.append(fp_sum)

        recalls = [value / num_gt if num_gt else 0.0 for value in cumulative_tp]
        precisions = [
            tp / max(tp + fp, 1) for tp, fp in zip(cumulative_tp, cumulative_fp)
        ]
        ap = compute_ap(recalls, precisions) if num_gt else 0.0
        if num_gt:
            aps.append(ap)

        total_tp += tp_sum
        total_fp += fp_sum
        total_gt += num_gt

        per_class[class_name] = {
            "ap": round(ap, 6),
            "num_ground_truth": num_gt,
            "num_predictions": len(class_preds),
            "true_positives": tp_sum,
            "false_positives": fp_sum,
            "recall": round(tp_sum / num_gt, 6) if num_gt else 0.0,
            "precision": round(tp_sum / max(tp_sum + fp_sum, 1), 6),
        }

    map_50 = sum(aps) / len(aps) if aps else 0.0
    performance_points = performance_score(map_50)

    return {
        "mAP@0.5": round(map_50, 6),
        "performance_points": performance_points,
        "iou_threshold": iou_threshold,
        "num_ground_truth_boxes": total_gt,
        "num_predictions": len(predictions),
        "micro_precision": round(total_tp / max(total_tp + total_fp, 1), 6),
        "micro_recall": round(total_tp / total_gt, 6) if total_gt else 0.0,
        "per_class": per_class,
    }


def performance_score(map_50: float) -> int:
    if map_50 < 0.10:
        return 0
    if map_50 < 0.20:
        return 5
    if map_50 < 0.35:
        return 10
    if map_50 < 0.50:
        return 15
    return 20


def main() -> None:
    args = parse_args()
    ground_truth = load_json(args.ground_truth)
    prediction_json = load_json(args.predictions)

    classes, image_info = validate_ground_truth(ground_truth)
    predictions = normalize_predictions(
        prediction_json,
        classes=classes,
        image_info=image_info,
        max_detections_per_image=args.max_detections_per_image,
        require_complete=not args.allow_missing_images,
    )

    result = evaluate(
        ground_truth=ground_truth,
        predictions=predictions,
        classes=classes,
        iou_threshold=args.iou_threshold,
    )

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
