import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, c1, ratio=16):
        super().__init__()
        c_ = max(8, c1 // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_, c1, 1, bias=False)
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.act(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, ratio=16, kernel_size=7):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channel_attention = None
        self.spatial_attention = None

    def forward(self, x):
        if self.channel_attention is None:  # 动态初始化
            c1 = x.shape[1]
            self.channel_attention = ChannelAttention(c1, self.ratio).to(x.device)
            self.spatial_attention = SpatialAttention(self.kernel_size).to(x.device)
        return self.spatial_attention(self.channel_attention(x))

