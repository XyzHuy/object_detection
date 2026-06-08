import torch.nn as nn
from .common import Conv, C2f, SPPF

try:
    import timm
except ImportError:  
    timm = None


class YOLOv8Backbone(nn.Module):
    def __init__(self, use_p2: bool = False):
        super().__init__()
        self.use_p2 = use_p2

        self.stem = Conv(3, 32, k = 3, s =2) # 320 -> 160

        self.stage1 = nn.Sequential(
            Conv(32, 64, k = 3, s =2),           # 160 -> 80
            C2f(64, 64, n = 2, shortcut = True)

        )
        self.stage2 = nn.Sequential(
            Conv(64,128, k = 3, s = 2),          # 80 -> 40
            C2f(128, 128, n = 2, shortcut = True) 

        )
        self.stage3 = nn.Sequential(
            Conv(128, 256, k=3, s=2),           # 40 -> 20
            C2f(256, 256, n=2, shortcut=True),
        )

        self.stage4 = nn.Sequential(
            Conv(256, 512, k=3, s=2),           # 20 -> 10
            C2f(512, 512, n=1, shortcut=True),
            SPPF(512, 512),
        )


    
    def forward(self, x):
        x = self.stem(x)
        p2 = self.stage1(x)

        p3 = self.stage2(p2)  # [B, 128, 40, 40]
        p4 = self.stage3(p3)  # [B, 256, 20, 20]
        p5 = self.stage4(p4)  # [B, 512, 10, 10]

        if self.use_p2:
            return p2, p3, p4, p5
        return p3, p4, p5


class ConvNeXtV2TinyBackbone(nn.Module):
    """
    Feature extractor ConvNeXt V2 Tiny pretrained ImageNet.
    Output khớp YOLOv8 neck:
      - P2/P3/P4/P5 stride 4/8/16/32, channel 64/128/256/512, hoặc
      - P3/P4/P5 stride 8/16/32, channel 128/256/512.
    """

    def __init__(
        self,
        model_name: str = "convnextv2_tiny",
        pretrained: bool = True,
        out_channels=None,
        use_p2: bool = False,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("Cần timm cho ConvNeXtV2TinyBackbone. Cài bằng: python3 -m pip install timm")

        self.use_p2 = use_p2
        out_indices = (0, 1, 2, 3) if use_p2 else (1, 2, 3)
        expected_reductions = (4, 8, 16, 32) if use_p2 else (8, 16, 32)
        if out_channels is None:
            out_channels = (64, 128, 256, 512) if use_p2 else (128, 256, 512)

        self.features = timm.create_model(
            model_name,
            features_only=True,
            pretrained=pretrained,
            out_indices=out_indices,
        )
        in_channels = self.features.feature_info.channels()
        reductions = self.features.feature_info.reduction()
        if tuple(reductions) != expected_reductions:
            raise ValueError(f"{model_name} trả về reductions {reductions}, cần {expected_reductions}.")

        self.adapters = nn.ModuleList(
            Conv(c1, c2, k=1, s=1) for c1, c2 in zip(in_channels, out_channels)
        )
        self.out_channels = tuple(out_channels)
        self.strides = tuple(reductions)

    def set_feature_extractor_trainable(self, trainable: bool):
        for param in self.features.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        feats = self.features(x)
        return tuple(adapter(feat) for adapter, feat in zip(self.adapters, feats))
