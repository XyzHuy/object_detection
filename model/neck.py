import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import Conv, C2f


class YOLOv8Neck(nn.Module):
    def __init__(self):
        super().__init__()


        #top down
        self.reduce_p5 = Conv(512, 256, k =1 , s = 1)
        self.c2f_p4 = C2f(256 + 256, 256, n =1)

        self.reduce_p4 = Conv(256, 128, k = 1, s = 1)
        self.c2f_p3 = C2f(128 + 128, 128 , n = 1)

        #bottom up
        self.down_p3 = Conv(128, 128, k = 3, s = 2)
        self.c2f_n4 = C2f(128 + 256, 256, n=1)

        self.down_p4 = Conv(256, 256, k=3, s=2)
        self.c2f_n5 = C2f(256 + 256, 512, n=1)
    

    def forward(self, p3, p4, p5):
        # top-down path
        p5_reduced = self.reduce_p5(p5)
        p5_up = F.interpolate(p5_reduced, scale_factor=2, mode="nearest")

        p4_fused = torch.cat([p5_up, p4], dim=1)
        p4_out = self.c2f_p4(p4_fused)

        p4_reduced = self.reduce_p4(p4_out)
        p4_up = F.interpolate(p4_reduced, scale_factor=2, mode="nearest")

        p3_fused = torch.cat([p4_up, p3], dim=1)
        p3_out = self.c2f_p3(p3_fused)

        # bottom-up path
        n4 = self.down_p3(p3_out)
        n4 = torch.cat([n4, p4_out], dim=1)
        n4 = self.c2f_n4(n4)

        n5 = self.down_p4(n4)
        n5 = torch.cat([n5, p5_reduced], dim=1)
        n5 = self.c2f_n5(n5)

        return p3_out, n4, n5