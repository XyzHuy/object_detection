import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Conv


class DFL(nn.Module):
    """
    Distribution Focal Layer.

    Input:
        x: [B, 4 * reg_max, N]
    Output:
        x: [B, 4, N]
    Ý tưởng:
        Mỗi cạnh bbox không dự đoán 1 số trực tiếp.
        Thay vào đó dự đoán distribution trên các bin: 0, 1, 2, ..., reg_max-1.
        Sau softmax, lấy expected value để ra khoảng cách 
    """

    def __init__(self, reg_max = 16):
        super().__init__()
        self.reg_max = reg_max

        self.proj = nn.Conv2d(reg_max, 1, kernel_size = 1, bias = False) # yêu cầu có shape [B*4, reg_max, N, 1]

        self.proj.weight.data[:] = torch.arange(
            reg_max, dtype = torch.float
        ).view(1, reg_max, 1, 1) # chuyển trọng số thành [0, 1, 2, ..., reg_max-1] để khi nhân với xác suất sẽ ra expected value

        for p in self.proj.parameters():
            p.requires_grad = False
    

    def forward(self, x):
        b,c,n = x.shape # [B, 4*reg_max, N]
        x = x.view(b,4,self.reg_max, n) # [B, 4, reg_max, N]

        x= x.softmax(dim = 2) # [B, 4, reg_max, N] (softmax trên dim reg_max để ra xác suất)
        x = x.permute(0, 1, 3, 2).contiguous() # [B, 4, N, reg_max]
        x = x.view(b * 4, n, self.reg_max)# [B*4, N, reg_max]
        x = x.permute(0, 2, 1).unsqueeze(-1) # [B*4, reg_max, N, 1]

        x = self.proj(x) # [B*4, 1, N, 1]

        x = x.squeeze(1).squeeze(-1)# [B*4, N]
        x = x.view(b, 4, n) # [B, 4, N]

        return x
    


def make_anchors(features, strides, offset = 0.5):
    """
    Tạo grid points cho từng feature map.

    features:
        list feature maps:
        [
            [B, C, H3, W3], 
            [B, C, H4, W4], 
            [B, C, H5, W5]
        ]

    strides:
        ví dụ [8, 16, 32]

    Return:
        anchor_points: [N, 2]
        stride_tensor: [N, 1]
    """
    anchor_points = []
    stride_tensor = []

    device = features[0].device
    dtype = features[0].dtype

    for feature, stride in zip(features, strides):
        _, _, h, w = feature.shape

        y  = torch.arange(h, device = device, dtype = dtype) + offset
        x = torch.arange(w, device = device, dtype = dtype) + offset

        yy, xx = torch.meshgrid(y, x, indexing = "ij")

        points = torch.stack([xx, yy], dim = -1).view(-1, 2)

        anchor_points.append(points)
        stride_tensor.append(
            torch.full((h*w, 1), stride, device = device, dtype = dtype)
        )
    
    anchor_points = torch.cat(anchor_points, dim = 0)
    stride_tensor = torch.cat(stride_tensor, dim = 0)

    return anchor_points, stride_tensor


def dist2bbox(distance, anchor_points, xywh = False):
    """
    Chuyển khoảng cách dự đoán thành bbox.

    distance: [B, 4, N]
    anchor_points: [N, 2]

    Return:
        bbox: [B, 4, N]
    """
    lt, rb = distance[:, 0:2, :], distance[:, 2:4, :]

    # anchor_points: [N, 2] -> [1, 2, N]
    anchor_points = anchor_points.T.unsqueeze(0)

    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb

    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim=1)

    return torch.cat((x1y1, x2y2), dim=1)


class YOLOv8DetectHead(nn.Module):
    """
    YOLOv8-like Detection Head.

    Input:
        features = [P3, P4, P5]

        P3: [B, 128, 40, 40]
        P4: [B, 256, 20, 20]
        P5: [B, 512, 10, 10]

    Training output:
        dict:
            boxes_raw: [B, 4 * reg_max, N]
            cls_logits: [B, num_classes, N]
            feats: original feature maps

    Inference output:
        pred: [B, 4 + num_classes, N]
             bbox đã decode + class probability
    """

    def __init__(
        self,
        num_classes,
        channels=(128, 256, 512),
        reg_max=16,
        strides=(8, 16, 32),
        head_depth: int = 2,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_outputs = num_classes + 4 * reg_max
        self.num_layers = len(channels)
        self.strides = strides
        head_depth = max(int(head_depth), 1)

        # Box regression branch
        self.box_heads = nn.ModuleList()

        # Classification branch
        self.cls_heads = nn.ModuleList()

        for c in channels:
            # box branch output = 4 * reg_max
            box_hidden = max(16, c // 4, 4 * reg_max)

            self.box_heads.append(
                nn.Sequential(
                    *[
                        Conv(c if idx == 0 else box_hidden, box_hidden, k=3, s=1)
                        for idx in range(head_depth)
                    ],
                    nn.Conv2d(box_hidden, 4 * reg_max, kernel_size=1),
                )
            )

            # cls branch output = num_classes
            cls_hidden = max(c, min(num_classes, 100))

            self.cls_heads.append(
                nn.Sequential(
                    *[
                        Conv(c if idx == 0 else cls_hidden, cls_hidden, k=3, s=1)
                        for idx in range(head_depth)
                    ],
                    nn.Conv2d(cls_hidden, num_classes, kernel_size=1),
                )
            )

        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()
        self.initialize_biases()

    def initialize_biases(self, image_size=640):
        for box_head, cls_head, stride in zip(self.box_heads, self.cls_heads, self.strides):
            box_head[-1].bias.data[:] = 1.0
            cls_head[-1].bias.data[: self.num_classes] = math.log(5 / self.num_classes / (image_size / stride) ** 2)

    def forward_head(self, features):
        """
        Convert feature maps thành dạng flatten.

        Return:
            box_raw: [B, 4 * reg_max, N]
            cls_logits: [B, num_classes, N]
        """

        batch_size = features[0].shape[0]

        box_outputs = []
        cls_outputs = []

        for i in range(self.num_layers):
            box = self.box_heads[i](features[i])
            cls = self.cls_heads[i](features[i])

            # [B, 4 * reg_max, H, W] -> [B, 4 * reg_max, H * W]
            box = box.view(batch_size, 4 * self.reg_max, -1)

            # [B, num_classes, H, W] -> [B, num_classes, H * W]
            cls = cls.view(batch_size, self.num_classes, -1)

            box_outputs.append(box)
            cls_outputs.append(cls)

        box_raw = torch.cat(box_outputs, dim=-1)
        cls_logits = torch.cat(cls_outputs, dim=-1)

        return box_raw, cls_logits

    def decode(self, box_raw, cls_logits, features, xywh=False):
        """
        Decode output khi inference.

        box_raw:
            [B, 4 * reg_max, N]

        cls_logits:
            [B, num_classes, N]

        Return:
            pred: [B, 4 + num_classes, N]
        """

        anchor_points, stride_tensor = make_anchors(
            features, self.strides, offset=0.5
        )

        # DFL:
        # [B, 4 * reg_max, N] -> [B, 4, N]
        pred_dist = self.dfl(box_raw)

        # Decode trên feature map coordinate
        pred_boxes = dist2bbox(
            pred_dist,
            anchor_points,
            xywh=xywh,
        )

        # Convert từ feature map coordinate về image pixel coordinate
        # stride_tensor: [N, 1] -> [1, 1, N]
        stride_tensor = stride_tensor.T.unsqueeze(0)
        pred_boxes = pred_boxes * stride_tensor

        pred_scores = cls_logits.sigmoid()

        pred = torch.cat([pred_boxes, pred_scores], dim=1)

        return pred

    def forward(self, features):
        box_raw, cls_logits = self.forward_head(features)

        outputs = {
            "boxes_raw": box_raw,
            "cls_logits": cls_logits,
            "feats": features,
        }

        if self.training:
            return outputs

        pred = self.decode(
            box_raw=box_raw,
            cls_logits=cls_logits,
            features=features,
            xywh=False,
        )

        return pred, outputs
