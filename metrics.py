from __future__ import annotations

from collections import defaultdict

import torch
from torchvision.ops import nms


def box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.shape[0], box2.shape[0]), device=box1.device)
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)
    lt = torch.maximum(box1[:, None, :2], box2[:, :2])
    rb = torch.minimum(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area1[:, None] + area2 - inter + eps)


def non_max_suppression(
    pred: torch.Tensor,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    max_det: int = 300,
) -> list[dict[str, torch.Tensor]]:
    if isinstance(pred, (tuple, list)):
        pred = pred[0]

    pred = pred.detach()
    boxes = pred[:, :4, :].permute(0, 2, 1)
    scores_all = pred[:, 4:, :].permute(0, 2, 1)
    results = []

    for image_boxes, image_scores in zip(boxes, scores_all):
        scores, labels = image_scores.max(dim=1)
        keep = scores >= conf_threshold
        image_boxes = image_boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if image_boxes.numel() == 0:
            results.append(
                {
                    "boxes": image_boxes.reshape(0, 4),
                    "scores": scores.reshape(0),
                    "labels": labels.reshape(0).long(),
                }
            )
            continue

        offsets = labels.to(image_boxes)[:, None] * 4096
        keep_idx = nms(image_boxes + offsets, scores, iou_threshold)[:max_det]
        results.append(
            {
                "boxes": image_boxes[keep_idx],
                "scores": scores[keep_idx],
                "labels": labels[keep_idx].long(),
            }
        )

    return results


def filter_predictions_by_score(
    predictions: list[dict[str, torch.Tensor]],
    score_threshold: float,
) -> list[dict[str, torch.Tensor]]:
    filtered = []
    for pred in predictions:
        keep = pred["scores"] >= score_threshold
        filtered.append(
            {
                "boxes": pred["boxes"][keep],
                "scores": pred["scores"][keep],
                "labels": pred["labels"][keep],
            }
        )
    return filtered


def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    ap = 0.0
    for idx in range(1, len(mrec)):
        if mrec[idx] != mrec[idx - 1]:
            ap += (mrec[idx] - mrec[idx - 1]) * mpre[idx]
    return ap


def detection_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict:
    gt_by_class = {class_idx: defaultdict(list) for class_idx in range(num_classes)}
    pred_by_class = {class_idx: [] for class_idx in range(num_classes)}

    for image_idx, target in enumerate(targets):
        boxes = target["boxes"].detach().cpu().float()
        labels = target["labels"].detach().cpu().long()
        for box, label in zip(boxes, labels):
            gt_by_class[int(label)][image_idx].append({"bbox": box, "matched": False})

    for image_idx, pred in enumerate(predictions):
        boxes = pred["boxes"].detach().cpu().float()
        scores = pred["scores"].detach().cpu().float()
        labels = pred["labels"].detach().cpu().long()
        for box, score, label in zip(boxes, scores, labels):
            pred_by_class[int(label)].append(
                {"image_idx": image_idx, "bbox": box, "score": float(score)}
            )

    aps = []
    per_class = {}
    total_tp = total_fp = total_gt = 0

    for class_idx in range(num_classes):
        class_gts = gt_by_class[class_idx]
        class_preds = sorted(pred_by_class[class_idx], key=lambda item: item["score"], reverse=True)
        num_gt = sum(len(items) for items in class_gts.values())
        tp, fp = [], []

        for pred in class_preds:
            candidates = class_gts.get(pred["image_idx"], [])
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(candidates):
                if gt["matched"]:
                    continue
                iou = box_iou_xyxy(pred["bbox"][None], gt["bbox"][None]).item()
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                candidates[best_idx]["matched"] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        cum_tp, cum_fp = [], []
        tp_sum = fp_sum = 0
        for t, f in zip(tp, fp):
            tp_sum += t
            fp_sum += f
            cum_tp.append(tp_sum)
            cum_fp.append(fp_sum)

        recalls = [value / num_gt if num_gt else 0.0 for value in cum_tp]
        precisions = [t / max(t + f, 1) for t, f in zip(cum_tp, cum_fp)]
        ap = compute_ap(recalls, precisions) if num_gt else 0.0
        if num_gt:
            aps.append(ap)

        total_tp += tp_sum
        total_fp += fp_sum
        total_gt += num_gt
        per_class[class_idx] = {
            "ap50": ap,
            "num_gt": num_gt,
            "num_predictions": len(class_preds),
            "precision": tp_sum / max(tp_sum + fp_sum, 1),
            "recall": tp_sum / num_gt if num_gt else 0.0,
        }

    return {
        "mAP50": sum(aps) / len(aps) if aps else 0.0,
        "precision": total_tp / max(total_tp + total_fp, 1),
        "recall": total_tp / total_gt if total_gt else 0.0,
        "num_predictions": sum(len(items) for items in pred_by_class.values()),
        "num_ground_truth": total_gt,
        "per_class": per_class,
    }
