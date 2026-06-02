import torch.nn as nn

from .backbone import ConvNeXtV2TinyBackbone, YOLOv8Backbone
from .neck import YOLOv8Neck
from .head import YOLOv8DetectHead


class YOLOv8Scratch(nn.Module):
    def __init__(
        self,
        num_classes,
        pretrained_backbone=True,
        backbone_name="convnextv2_tiny",
        use_pretrained_backbone=True,
        use_p2=False,
        neck_depth=1,
        head_depth=2,
    ):
        super().__init__()
        self.use_p2 = use_p2
        head_channels = (64, 128, 256, 512) if use_p2 else (128, 256, 512)
        head_strides = (4, 8, 16, 32) if use_p2 else (8, 16, 32)

        if use_pretrained_backbone:
            self.backbone = ConvNeXtV2TinyBackbone(
                model_name=backbone_name,
                pretrained=pretrained_backbone,
                use_p2=use_p2,
            )
        else:
            self.backbone = YOLOv8Backbone(use_p2=use_p2)
        self.neck = YOLOv8Neck(use_p2=use_p2, c2f_depth=neck_depth)

        self.head = YOLOv8DetectHead(
            num_classes=num_classes,
            channels=head_channels,
            reg_max=16,
            strides=head_strides,
            head_depth=head_depth,
        )

    def forward(self, x):
        backbone_features = self.backbone(x)
        features = self.neck(*backbone_features)
        return self.head(features)

    def forward_features(self, x):
        backbone_features = self.backbone(x)
        return list(self.neck(*backbone_features))

    def loss_outputs(self, x):
        features = self.forward_features(x)
        box_raw, cls_logits = self.head.forward_head(features)
        return {
            "boxes_raw": box_raw,
            "cls_logits": cls_logits,
            "feats": features,
        }


YOLOv8 = YOLOv8Scratch
