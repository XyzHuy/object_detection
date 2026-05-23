import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p = None, groups = 1):
        super().__init__()
        if p is None:
            p = k // 2
        
        self.conv = nn.Conv2d(c1, c2, kernel_size = k, stride = s, padding = p, groups = groups, bias = False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace = True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut = True):
        super().__init__()
        hidden = c2 // 2
        self.cv1 = Conv(c1, hidden, k = 3, s = 1)
        self.cv2 = Conv(hidden, c2 , k = 3, s = 1)
        self.use_shortcut = shortcut and c1 == c2
    
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        if self.use_shortcut:
            return x+y
        
        return y
    


class C2f(nn.Module):
    def __init__(self, c1, c2, n = 1, shortcut = False):
        super().__init__()
        hidden = c2 // 2

        self.cv1 = Conv(c1, hidden *2, k =1, s = 1)
        self.blocks = nn.ModuleList([
            Bottleneck(hidden, hidden, shortcut = shortcut)
            for _ in range(n)
        ])
        self.cv2 = Conv(hidden*(2+n), c2, k=1, s=1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2,dim =1))
        for block in self.blocks:
            y.append(block(y[-1]))
        
        return self.cv2(torch.cat(y, dim = 1))
    

class SPPF(nn.Module):
    def __init__(self, c1, c2, k =5):
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, k = 1, s= 1)
        self.cv2 = Conv(hidden*4, c2, k = 1, s = 1 )
        self.pool = nn.MaxPool2d(kernel_size = k, stride = 1, padding = k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x,y1,y2,y3], dim = 1))
