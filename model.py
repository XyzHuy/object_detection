"""
Baseline Object Detection Model
Backbone : MobileNetV2 (pretrained, frozen early layers)
Head     : Lightweight multi-scale detection head (SSD-style)
Loss     : Focal loss (cls) + Smooth-L1 (reg)
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.ops import box_iou
from typing import List, Dict, Tuple
import math


# ---------------------------------------------------------------------------
# Anchor generator
# ---------------------------------------------------------------------------

class AnchorGenerator(nn.Module):
    def __init__(self, sizes=((32,), (64,), (128,), (256,)),
                 aspect_ratios=((0.5, 1.0, 2.0),) * 4):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

    def generate_anchors(self, scale, aspect_ratios, device):
        anchors = []
        for ar in aspect_ratios:
            w = scale * math.sqrt(ar)
            h = scale / math.sqrt(ar)
            anchors.append([-w/2, -h/2, w/2, h/2])
        return torch.tensor(anchors, dtype=torch.float32, device=device)

    def forward(self, feature_maps: List[torch.Tensor], image_size: Tuple[int, int]):
        all_anchors = []
        H_img, W_img = image_size
        for i, feat in enumerate(feature_maps):
            H_f, W_f = feat.shape[-2:]
            stride_h = H_img / H_f
            stride_w = W_img / W_f

            base = self.generate_anchors(self.sizes[i][0], self.aspect_ratios[i], feat.device)
            # center points (H_f * W_f grid)
            cy = (torch.arange(H_f, device=feat.device).float() + 0.5) * stride_h
            cx = (torch.arange(W_f, device=feat.device).float() + 0.5) * stride_w
            grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")
            grid = torch.stack([grid_x, grid_y, grid_x, grid_y], dim=-1).reshape(-1, 4)  # (H*W, 4)
            anchors_per_map = grid.unsqueeze(1) + base.unsqueeze(0)   # (H*W, A, 4)
            anchors_per_map = anchors_per_map.reshape(-1, 4)           # (H*W*A, 4)
            all_anchors.append(anchors_per_map)

        return torch.cat(all_anchors, dim=0)  # (N_anchors, 4)  xyxy format


# ---------------------------------------------------------------------------
# Detection Head (shared conv → cls + reg per scale)
# ---------------------------------------------------------------------------

class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(256, num_anchors * num_classes, 1)
        self.reg_head = nn.Conv2d(256, num_anchors * 4, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # focal loss bias init
        prior = 0.01
        nn.init.constant_(self.cls_head.bias, -math.log((1 - prior) / prior))

    def forward(self, features: List[torch.Tensor]):
        cls_logits, bbox_preds = [], []
        for feat in features:
            x = self.shared(feat)
            cls = self.cls_head(x)   # (B, A*C, H, W)
            reg = self.reg_head(x)   # (B, A*4, H, W)

            B, _, H, W = cls.shape
            cls = cls.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            reg = reg.permute(0, 2, 3, 1).reshape(B, -1, 4)

            cls_logits.append(cls)
            bbox_preds.append(reg)

        return torch.cat(cls_logits, dim=1), torch.cat(bbox_preds, dim=1)  # (B, N_anc, ...)


# ---------------------------------------------------------------------------
# FPN-lite: lateral + output convs for 4 scales from MobileNetV2
# ---------------------------------------------------------------------------

class FPNLite(nn.Module):
    """Minimal top-down FPN for MobileNetV2 feature maps."""
    def __init__(self, in_channels_list: List[int], out_channels: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        self.output = nn.ModuleList([
            nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True))
            for _ in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # bottom-up already done; apply lateral convs
        lat = [self.lateral[i](f) for i, f in enumerate(features)]

        # top-down fusion
        for i in range(len(lat) - 1, 0, -1):
            upsampled = nn.functional.interpolate(lat[i], size=lat[i-1].shape[-2:], mode="nearest")
            lat[i-1] = lat[i-1] + upsampled

        return [self.output[i](lat[i]) for i in range(len(lat))]


# ---------------------------------------------------------------------------
# Full detector
# ---------------------------------------------------------------------------

class SimpleDetector(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        # ---- Backbone ----
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        features = backbone.features

        # Extract 4 feature maps at different strides
        # stride 8  → features[0:7]
        # stride 16 → features[7:14]
        # stride 32 → features[14:19]
        self.layer1 = features[0:7]   # out: 32ch,  stride 8
        self.layer2 = features[7:14]  # out: 96ch,  stride 16
        self.layer3 = features[14:]   # out: 1280ch, stride 32

        # Extra scale (stride 64)
        self.extra = nn.Sequential(
            nn.Conv2d(1280, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Freeze first 4 layers of backbone to speed up training
        for param in list(self.layer1.parameters()):
            param.requires_grad = False

        # ---- FPN ----
        self.fpn = FPNLite(in_channels_list=[32, 96, 1280, 256], out_channels=128)

        # ---- Anchors ----
        num_anchors = 3
        self.anchor_gen = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 4,
        )

        # ---- Head ----
        self.head = DetectionHead(
            in_channels=128,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

    def forward(self, images: List[torch.Tensor]):
        # Sau letterbox tất cả ảnh cùng size → stack bình thường
        x = torch.stack(images)            # (B, 3, H, W)
        H, W = x.shape[-2:]               # đọc size thực, không hardcode

        f1 = self.layer1(x)                # stride 8
        f2 = self.layer2(f1)               # stride 16
        f3 = self.layer3(f2)               # stride 32
        f4 = self.extra(f3)                # stride 64

        fpn_feats = self.fpn([f1, f2, f3, f4])

        cls_logits, bbox_preds = self.head(fpn_feats)   # (B, N, C), (B, N, 4)

        anchors = self.anchor_gen(fpn_feats, (H, W))    # (N, 4)
        # Clamp anchors to image bounds
        anchors[:, [0, 2]] = anchors[:, [0, 2]].clamp(0, W)
        anchors[:, [1, 3]] = anchors[:, [1, 3]].clamp(0, H)

        return cls_logits, bbox_preds, anchors


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Binary focal loss per class."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = targets * prob + (1 - targets) * (1 - prob)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    loss = alpha_t * (1 - p_t) ** gamma * bce
    return loss.sum()


def encode_boxes(proposals: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    """Delta encoding (anchor → gt), same as Faster-RCNN."""
    wa = proposals[:, 2] - proposals[:, 0]
    ha = proposals[:, 3] - proposals[:, 1]
    cxa = (proposals[:, 0] + proposals[:, 2]) / 2
    cya = (proposals[:, 1] + proposals[:, 3]) / 2

    wg = references[:, 2] - references[:, 0]
    hg = references[:, 3] - references[:, 1]
    cxg = (references[:, 0] + references[:, 2]) / 2
    cyg = (references[:, 1] + references[:, 3]) / 2

    dx = (cxg - cxa) / wa
    dy = (cyg - cya) / ha
    dw = torch.log(wg / wa + 1e-6)
    dh = torch.log(hg / ha + 1e-6)
    return torch.stack([dx, dy, dw, dh], dim=1)


def detection_loss(cls_logits: torch.Tensor,  # (B, N, C)
                   bbox_preds: torch.Tensor,   # (B, N, 4)
                   anchors: torch.Tensor,       # (N, 4)
                   targets: List[Dict],
                   pos_iou: float = 0.5,
                   neg_iou: float = 0.4,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:

    B, N, C = cls_logits.shape
    device = cls_logits.device

    total_cls = torch.tensor(0.0, device=device)
    total_reg = torch.tensor(0.0, device=device)
    num_pos = 0

    for b in range(B):
        gt_boxes = targets[b]["boxes"].to(device)   # (M, 4)
        gt_labels = targets[b]["labels"].to(device) # (M,)

        if gt_boxes.numel() == 0:
            # No GT → all negatives
            cls_t = torch.zeros(N, C, device=device)
            total_cls += focal_loss(cls_logits[b], cls_t)
            continue

        iou = box_iou(anchors, gt_boxes)            # (N, M)
        max_iou, best_gt = iou.max(dim=1)           # (N,)

        # Assign
        pos_mask = max_iou >= pos_iou
        neg_mask = max_iou < neg_iou
        ignore_mask = ~(pos_mask | neg_mask)

        # Force assign each GT to its best anchor
        best_anchor_per_gt = iou.argmax(dim=0)
        pos_mask[best_anchor_per_gt] = True
        ignore_mask[best_anchor_per_gt] = False

        num_pos_b = pos_mask.sum().item()
        num_pos += num_pos_b

        # --- Classification target ---
        cls_t = torch.zeros(N, C, device=device)
        if num_pos_b > 0:
            assigned_labels = gt_labels[best_gt[pos_mask]]   # (num_pos,)
            cls_t[pos_mask, assigned_labels] = 1.0

        valid_mask = ~ignore_mask
        total_cls += focal_loss(cls_logits[b][valid_mask], cls_t[valid_mask])

        # --- Regression target (only positives) ---
        if num_pos_b > 0:
            assigned_boxes = gt_boxes[best_gt[pos_mask]]     # (num_pos, 4)
            pos_anchors = anchors[pos_mask]                   # (num_pos, 4)
            deltas_target = encode_boxes(pos_anchors, assigned_boxes)
            deltas_pred = bbox_preds[b][pos_mask]
            total_reg += nn.functional.smooth_l1_loss(deltas_pred, deltas_target, reduction="sum", beta=0.1)

    norm = max(num_pos, 1)
    return total_cls / norm, total_reg / norm


# ---------------------------------------------------------------------------
# Decode boxes (inference)
# ---------------------------------------------------------------------------

def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    cxa = (anchors[:, 0] + anchors[:, 2]) / 2
    cya = (anchors[:, 1] + anchors[:, 3]) / 2

    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    cx = dx * wa + cxa
    cy = dy * ha + cya
    w = torch.exp(dw.clamp(max=4)) * wa
    h = torch.exp(dh.clamp(max=4)) * ha

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)