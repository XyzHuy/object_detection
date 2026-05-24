import torch.nn as nn

from .backbone import CSPDarknetBackbone, YOLOv8Backbone
from .local_jepa import LocalJEPABranch
from .neck import YOLOv8Neck
from .head import YOLOv8DetectHead


class YOLOv8Scratch(nn.Module):
    def __init__(
        self,
        num_classes,
        pretrained_backbone=True,
        backbone_name="cspdarknet53",
        use_cspdarknet=True,
        use_local_jepa=False,
    ):
        super().__init__()
        self.use_local_jepa = use_local_jepa

        if use_cspdarknet:
            self.backbone = CSPDarknetBackbone(
                model_name=backbone_name,
                pretrained=pretrained_backbone,
            )
        else:
            self.backbone = YOLOv8Backbone()
        self.neck = YOLOv8Neck()
        self.local_jepa = LocalJEPABranch() if use_local_jepa else None

        self.head = YOLOv8DetectHead(
            num_classes=num_classes,
            channels=(128, 256, 512),
            reg_max=16,
            strides=(8, 16, 32),
        )

    def forward(self, x):
        backbone_features = self.backbone(x)
        features = self.neck(*backbone_features)
        out = self.head(features)
        if self.training and self.use_local_jepa and self.local_jepa is not None:
            out["local_jepa"] = self.local_jepa(backbone_features)
        return out

    def forward_features(self, x):
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)
        return [n3, n4, n5]

    def loss_outputs(self, x):
        features = self.forward_features(x)
        box_raw, cls_logits = self.head.forward_head(features)
        return {
            "boxes_raw": box_raw,
            "cls_logits": cls_logits,
            "feats": features,
        }


YOLOv8 = YOLOv8Scratch
