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

        self.stem = Conv(3, 32, k = 3, s =2) #320 -> 160

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


class CSPDarknetBackbone(nn.Module):
    """
    ImageNet-pretrained CSPDarkNet feature extractor with the same output
    contract as the existing YOLOv8 backbone: P3/P4/P5 at strides 8/16/32
    and channels 128/256/512.
    """

    def __init__(
        self,
        model_name: str = "cspdarknet53",
        pretrained: bool = True,
        out_channels=(128, 256, 512),
        use_p2: bool = False,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for CSPDarknetBackbone. Install it with: python3 -m pip install timm")

        self.use_p2 = use_p2
        out_indices = (2, 3, 4, 5) if use_p2 else (3, 4, 5)
        expected_reductions = (4, 8, 16, 32) if use_p2 else (8, 16, 32)
        if use_p2 and tuple(out_channels) == (128, 256, 512):
            out_channels = (64, 128, 256, 512)

        self.features = timm.create_model(
            model_name,
            features_only=True,
            pretrained=pretrained,
            out_indices=out_indices,
        )
        in_channels = self.features.feature_info.channels()
        reductions = self.features.feature_info.reduction()
        if tuple(reductions) != expected_reductions:
            raise ValueError(f"{model_name} returned reductions {reductions}, expected {expected_reductions}.")

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
