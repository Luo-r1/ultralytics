import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, reduction=32):
        super().__init__()
        self.reduction = reduction
        self.conv1 = None
        self.bn1 = None
        self.act = nn.ReLU(inplace=True)
        self.conv_h = None
        self.conv_w = None
        self.inited = False  # 延迟初始化标志

    def forward(self, x):
        n, c, h, w = x.size()

        if not self.inited:  # 第一次 forward 的时候才初始化
            c_ = max(8, c // self.reduction)
            self.conv1 = nn.Conv2d(c, c_, 1, stride=1).to(x.device)
            self.bn1 = nn.BatchNorm2d(c_).to(x.device)
            self.conv_h = nn.Conv2d(c_, c, 1, stride=1).to(x.device)
            self.conv_w = nn.Conv2d(c_, c, 1, stride=1).to(x.device)
            self.inited = True

        # 高度池化 [N,C,H,1]，宽度池化 [N,C,1,W]
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w))
        x_w = x_w.permute(0, 1, 3, 2)

        # 拼接
        y = torch.cat([x_h, x_w], dim=2)  # [N,C,H+W,1]
        y = self.act(self.bn1(self.conv1(y)))

        # 拆分
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 注意力
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return x * a_h * a_w


