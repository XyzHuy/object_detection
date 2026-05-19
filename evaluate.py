"""
Evaluation: tính mAP@0.5 và mAP@0.5:0.95 đơn giản.

Usage:
    python evaluate.py --data_root /path/to/dataset --checkpoint checkpoints/best.pth --num_classes 10
"""

import argparse
from collections import defaultdict

import torch
import torchvision
from torchvision.ops import batched_nms, box_iou

from model import SimpleDetector, decode_boxes
from dataloader import build_dataloader


# ---------------------------------------------------------------------------
# Arg parse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Evaluate SimpleDetector")
    p.add_argument("--data_root",   required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--img_size",    type=int, default=320)
    p.add_argument("--split",       default="val")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--conf_thresh", type=float, default=0.05)
    p.add_argument("--nms_thresh",  type=float, default=0.5)
    p.add_argument("--iou_thresh",  type=float, default=0.5, help="IoU threshold for mAP")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Post-process: decode + NMS
# ---------------------------------------------------------------------------

@torch.no_grad()
def postprocess(cls_logits, bbox_preds, anchors, conf_thresh, nms_thresh):
    """
    Returns list of dicts per image:
        {"boxes": (N,4), "scores": (N,), "labels": (N,)}
    """
    B = cls_logits.shape[0]
    results = []

    for b in range(B):
        scores_all = cls_logits[b].sigmoid()   # (N_anc, C)
        boxes_all  = decode_boxes(anchors, bbox_preds[b])  # (N_anc, 4)

        # Keep only anchors whose max class score > threshold
        max_scores, labels = scores_all.max(dim=1)
        keep = max_scores > conf_thresh

        if keep.sum() == 0:
            results.append({"boxes": torch.zeros((0,4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)})
            continue

        boxes_f  = boxes_all[keep]
        scores_f = max_scores[keep]
        labels_f = labels[keep]

        # NMS per class
        keep_nms = batched_nms(boxes_f, scores_f, labels_f, nms_thresh)
        results.append({
            "boxes":  boxes_f[keep_nms].cpu(),
            "scores": scores_f[keep_nms].cpu(),
            "labels": labels_f[keep_nms].cpu(),
        })
    return results


# ---------------------------------------------------------------------------
# AP computation (11-point interpolation)
# ---------------------------------------------------------------------------

def compute_ap(recalls, precisions):
    """Compute AP using 11-point interpolation."""
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        prec_at_t = [p for r, p in zip(recalls, precisions) if r >= t]
        ap += max(prec_at_t) if prec_at_t else 0.0
    return ap / 11.0


def compute_map(all_preds, all_gts, num_classes, iou_thresh=0.5):
    """
    all_preds: list of {"boxes":(N,4), "scores":(N,), "labels":(N,)}
    all_gts  : list of {"boxes":(M,4), "labels":(M,)}
    Returns: mAP (float), per_class_ap (dict)
    """
    # Collect per-class detections and GTs
    class_dets = defaultdict(list)   # cls → [(score, tp/fp), ...]
    class_ngt  = defaultdict(int)    # cls → total GT count

    for img_idx, (pred, gt) in enumerate(zip(all_preds, all_gts)):
        gt_boxes  = gt["boxes"]
        gt_labels = gt["labels"]
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for cls in range(num_classes):
            class_ngt[cls] += (gt_labels == cls).sum().item()

        # Sort preds by score descending
        if len(pred["scores"]) == 0:
            continue
        order = pred["scores"].argsort(descending=True)
        p_boxes  = pred["boxes"][order]
        p_scores = pred["scores"][order]
        p_labels = pred["labels"][order]

        for i in range(len(p_scores)):
            cls  = p_labels[i].item()
            box  = p_boxes[i].unsqueeze(0)
            scr  = p_scores[i].item()

            gt_cls_mask = (gt_labels == cls)
            if gt_cls_mask.sum() == 0:
                class_dets[cls].append((scr, 0))
                continue

            gt_cls_boxes = gt_boxes[gt_cls_mask]
            gt_cls_idx   = gt_cls_mask.nonzero(as_tuple=False).squeeze(1)

            iou = box_iou(box, gt_cls_boxes)[0]  # (M_cls,)
            best_iou, best_j = iou.max(dim=0) if len(iou) > 0 else (torch.tensor(0.0), torch.tensor(0))

            if best_iou >= iou_thresh and not gt_matched[gt_cls_idx[best_j]]:
                gt_matched[gt_cls_idx[best_j]] = True
                class_dets[cls].append((scr, 1))   # TP
            else:
                class_dets[cls].append((scr, 0))   # FP

    # Compute AP per class
    per_class_ap = {}
    for cls in range(num_classes):
        dets = sorted(class_dets[cls], key=lambda x: -x[0])
        ngt  = class_ngt[cls]
        if ngt == 0:
            per_class_ap[cls] = float("nan")
            continue

        tp_cum = 0; fp_cum = 0
        recalls = []; precisions = []
        for scr, tp in dets:
            if tp: tp_cum += 1
            else:  fp_cum += 1
            recalls.append(tp_cum / ngt)
            precisions.append(tp_cum / (tp_cum + fp_cum))

        per_class_ap[cls] = compute_ap(recalls, precisions)

    valid_aps = [v for v in per_class_ap.values() if not torch.isnan(torch.tensor(v))]
    mAP = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0
    return mAP, per_class_ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SimpleDetector(num_classes=args.num_classes).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Dataloader (no normalization needed for gt, but keep same pipeline)
    loader = build_dataloader(
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True,
        img_size=args.img_size,
    )

    all_preds, all_gts = [], []

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            cls_logits, bbox_preds, anchors = model(images)

            preds = postprocess(cls_logits, bbox_preds, anchors,
                                args.conf_thresh, args.nms_thresh)
            all_preds.extend(preds)
            all_gts.extend([{k: v for k, v in t.items()} for t in targets])

    mAP, per_cls = compute_map(all_preds, all_gts, args.num_classes, iou_thresh=args.iou_thresh)
    print(f"\n{'='*40}")
    print(f"mAP@{args.iou_thresh:.2f} = {mAP*100:.2f}%")
    print(f"{'='*40}")
    print(f"{'Class':>8} | AP")
    for cls, ap in per_cls.items():
        marker = "  (no GT)" if torch.isnan(torch.tensor(ap)) else ""
        ap_str = f"{ap*100:.2f}%" if not torch.isnan(torch.tensor(ap)) else "  N/A"
        print(f"  cls {cls:3d}  | {ap_str}{marker}")


if __name__ == "__main__":
    main()