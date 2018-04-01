# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.size()
        g = self.groups
        h = x.view(n, g, c/g, h, w).permute(0, 2, 1, 3, 4)
        return h.contiguous().view(n, c, h, w)


class BottleNeck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(BottleNeck, self).__init__()
        self.stride = stride

        mid_planes = out_planes / 4
        g = 1 if in_planes == 24 else groups

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1,
                               groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3,
                               stride=stride, padding=1,
                               groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1,
                               groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        res = self.shortcut(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.shuffle1(h)
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        if self.stride == 2:
            h = F.relu(torch.caa([h, res], 1))
        else:
            h = F.relu(h + res)
        return h


class ShuffleNet(nn.Module):
    def __init__(self, n_class, out_planes=[240, 480, 960],
                 num_blocks=[4, 8, 4], groups=3):
        super(ShuffleNet, self).__init__()
        out_planes = out_planes
        num_blocks = num_blocks
        groups = groups

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[2], n_class)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(BottleNeck(self.in_planes, out_planes-cat_planes,
                                     stride=stride, groups=groups))
            self.in_planes = out_planes

        return nn.Sequential(*layers)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.avg_pool2d(h, 4)
        h = h.view(h.size(0), -1)
        h = self.linear(h)
        return h
