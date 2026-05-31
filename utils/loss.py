from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model.head import dist2bbox, make_anchors


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)

    lt = torch.maximum(box1[:, None, :2], box2[:, :2])
    rb = torch.minimum(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area1[:, None] + area2 - inter + eps)


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x1 = torch.maximum(box1[:, 0], box2[:, 0])
    y1 = torch.maximum(box1[:, 1], box2[:, 1])
    x2 = torch.minimum(box1[:, 2], box2[:, 2])
    y2 = torch.minimum(box1[:, 3], box2[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    w1 = (box1[:, 2] - box1[:, 0]).clamp(min=eps)
    h1 = (box1[:, 3] - box1[:, 1]).clamp(min=eps)
    w2 = (box2[:, 2] - box2[:, 0]).clamp(min=eps)
    h2 = (box2[:, 3] - box2[:, 1]).clamp(min=eps)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    c1 = (box1[:, :2] + box1[:, 2:]) / 2
    c2 = (box2[:, :2] + box2[:, 2:]) / 2
    rho2 = ((c1 - c2) ** 2).sum(dim=1)

    enclose_lt = torch.minimum(box1[:, :2], box2[:, :2])
    enclose_rb = torch.maximum(box1[:, 2:], box2[:, 2:])
    c2_diag = ((enclose_rb - enclose_lt) ** 2).sum(dim=1).clamp(min=eps)

    v = (4 / torch.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2_diag + alpha * v)


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int) -> torch.Tensor:
    x1y1, x2y2 = bbox[..., :2], bbox[..., 2:]
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), dim=-1).clamp(0, reg_max - 0.01)


@dataclass
class YOLOv8LossItems:
    loss: torch.Tensor
    box_loss: torch.Tensor
    cls_loss: torch.Tensor
    dfl_loss: torch.Tensor


class YOLOv8Loss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        strides=(8, 16, 32),
        reg_max: int = 16,
        topk: int = 10,
        alpha: float = 0.5,
        beta: float = 6.0,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        class_weights: torch.Tensor | list[float] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        if class_weights is None:
            weights = torch.ones(num_classes, dtype=torch.float32)
        else:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.numel() != num_classes:
                raise ValueError(f"Expected {num_classes} class weights, got {weights.numel()}")
        self.register_buffer("class_weights", weights.view(1, 1, num_classes))
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, outputs: dict, targets: list[dict]) -> YOLOv8LossItems:
        feats = outputs["feats"]
        box_raw = outputs["boxes_raw"]
        cls_logits = outputs["cls_logits"]
        device = cls_logits.device

        anchor_points, stride_tensor = make_anchors(feats, self.strides, offset=0.5)
        stride_tensor = stride_tensor.to(device)
        anchor_points = anchor_points.to(device)

        pred_dist = outputs.get("dfl")
        if pred_dist is None:
            pred_dist = self._dfl_decode(box_raw)

        pred_boxes_grid = dist2bbox(pred_dist, anchor_points, xywh=False).permute(0, 2, 1)
        pred_boxes = pred_boxes_grid * stride_tensor.view(1, -1, 1)
        pred_scores = cls_logits.permute(0, 2, 1).sigmoid()
        cls_logits = cls_logits.permute(0, 2, 1)

        target_boxes, target_scores, fg_mask = self.assign(
            pred_scores.detach(),
            pred_boxes.detach(),
            anchor_points * stride_tensor,
            targets,
            device,
        )
        target_scores_sum = target_scores.sum().clamp(min=1.0)

        cls_weight = torch.where(target_scores > 0, self.class_weights.to(device), 1.0)
        cls_loss_raw = self.bce(cls_logits, target_scores)
        cls_loss = (cls_loss_raw * cls_weight).sum() / target_scores_sum

        if fg_mask.any():
            weight = target_scores.sum(dim=-1)[fg_mask]
            box_loss = ((1.0 - bbox_ciou(pred_boxes[fg_mask], target_boxes[fg_mask])) * weight).sum()
            box_loss = box_loss / target_scores_sum

            target_ltrb = bbox2dist(
                anchor_points,
                target_boxes / stride_tensor.view(1, -1, 1),
                self.reg_max - 1,
            )
            dfl_loss = self.distribution_focal_loss(box_raw, target_ltrb, fg_mask, weight)
            dfl_loss = dfl_loss / target_scores_sum
        else:
            box_loss = box_raw.sum() * 0.0
            dfl_loss = box_raw.sum() * 0.0

        total = (
            self.box_gain * box_loss
            + self.cls_gain * cls_loss
            + self.dfl_gain * dfl_loss
        )
        return YOLOv8LossItems(
            total,
            box_loss.detach(),
            cls_loss.detach(),
            dfl_loss.detach(),
        )

    def _dfl_decode(self, box_raw: torch.Tensor) -> torch.Tensor:
        b, _, n = box_raw.shape
        proj = torch.arange(self.reg_max, device=box_raw.device, dtype=box_raw.dtype)
        pred = box_raw.view(b, 4, self.reg_max, n).softmax(dim=2)
        return (pred * proj.view(1, 1, self.reg_max, 1)).sum(dim=2)

    @torch.no_grad()
    def assign(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        anchor_points_img: torch.Tensor,
        targets: list[dict],
        device: torch.device,
    ):
        bsz, num_anchors, _ = pred_scores.shape
        target_boxes = torch.zeros((bsz, num_anchors, 4), device=device)
        target_scores = torch.zeros((bsz, num_anchors, self.num_classes), device=device)
        fg_mask = torch.zeros((bsz, num_anchors), dtype=torch.bool, device=device)

        for batch_idx, target in enumerate(targets):
            gt_boxes = target["boxes"].to(device).float()
            gt_labels = target["labels"].to(device).long()
            valid = (gt_boxes[:, 2] > gt_boxes[:, 0]) & (gt_boxes[:, 3] > gt_boxes[:, 1])
            gt_boxes = gt_boxes[valid]
            gt_labels = gt_labels[valid]
            if gt_boxes.numel() == 0:
                continue

            in_gt = self._anchors_in_boxes(anchor_points_img, gt_boxes)
            ious = box_iou(pred_boxes[batch_idx], gt_boxes).clamp(min=0)
            cls_scores = pred_scores[batch_idx][:, gt_labels].clamp(min=1e-9)
            metrics = cls_scores.pow(self.alpha) * ious.pow(self.beta)
            metrics = metrics.masked_fill(~in_gt, 0.0)

            assigned_metric = torch.zeros(num_anchors, device=device)
            assigned_gt = torch.full((num_anchors,), -1, dtype=torch.long, device=device)

            for gt_idx in range(gt_boxes.shape[0]):
                metric = metrics[:, gt_idx]
                pos = metric > 0
                if not pos.any():
                    center = (gt_boxes[gt_idx, :2] + gt_boxes[gt_idx, 2:]) / 2
                    nearest = torch.cdist(anchor_points_img, center[None]).argmin()
                    pos_idx = nearest.view(1)
                else:
                    candidate = torch.where(pos)[0]
                    k = min(self.topk, candidate.numel())
                    pos_idx = candidate[metric[candidate].topk(k).indices]

                better = metrics[pos_idx, gt_idx] >= assigned_metric[pos_idx]
                selected = pos_idx[better]
                assigned_metric[selected] = metrics[selected, gt_idx]
                assigned_gt[selected] = gt_idx

            keep = assigned_gt >= 0
            if keep.any():
                matched_gt = assigned_gt[keep]
                fg_mask[batch_idx, keep] = True
                target_boxes[batch_idx, keep] = gt_boxes[matched_gt]
                target_scores[batch_idx, keep, gt_labels[matched_gt]] = 1.0

        return target_boxes, target_scores, fg_mask

    @staticmethod
    def _anchors_in_boxes(anchor_points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        x, y = anchor_points[:, 0:1], anchor_points[:, 1:2]
        return (x >= boxes[:, 0]) & (x <= boxes[:, 2]) & (y >= boxes[:, 1]) & (y <= boxes[:, 3])

    def distribution_focal_loss(
        self,
        box_raw: torch.Tensor,
        target_ltrb: torch.Tensor,
        fg_mask: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        pred = box_raw.permute(0, 2, 1).reshape(-1, 4, self.reg_max)[fg_mask.reshape(-1)]
        target = target_ltrb.reshape(-1, 4)[fg_mask.reshape(-1)]
        tl = target.long()
        tr = (tl + 1).clamp(max=self.reg_max - 1)
        wl = tr.float() - target
        wr = 1.0 - wl

        loss = (
            F.cross_entropy(pred.reshape(-1, self.reg_max), tl.reshape(-1), reduction="none").view(-1, 4) * wl
            + F.cross_entropy(pred.reshape(-1, self.reg_max), tr.reshape(-1), reduction="none").view(-1, 4) * wr
        ).mean(dim=1)
        return (loss * weight).sum()
