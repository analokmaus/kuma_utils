import torch
from torch import nn
import torch.nn.functional as F
from .pooling import ChannelPool


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=12):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.pool = ChannelPool()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(
            kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(self.pool(x))
        return self.sigmoid(x)


class CBAM2d(nn.Module):

    def __init__(self, in_planes, kernel_size=7):
        super().__init__()

        self.ch_attn = ChannelAttention(in_planes)
        self.sp_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        # x: bs x ch x w x h
        x = self.ch_attn(x) * x
        x = self.sp_attn(x) * x
        return x


class Attention3d(nn.Module):

    def __init__(self, in_planes, return_mask=False):
        super().__init__()

        self.ch_pool = ChannelPool(dim=2, concat=False)
        self.ch_attn = ChannelAttention(in_planes, ratio=4)
        self.return_mask = return_mask

    def forward(self, x):
        bs, n, ch, w, h = x.shape
        mask = self.ch_attn(self.ch_pool(x)[1].view(bs, n, w, h))
        if self.return_mask:
            return mask.view(bs, n, 1, 1, 1)
        else:
            x = x * mask.view(bs, n, 1, 1, 1)
            return x
