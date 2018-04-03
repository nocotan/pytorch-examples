# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, n_hidden=100, bottom_width=4, ch=512, wscale=0.02):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width

        self.l0 = nn.Linear(self.n_hidden, bottom_width * bottom_width * ch)
        self.dc1 = nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1, bias=False)
        self.dc2 = nn.ConvTranspose2d(ch // 2, ch // 4, 4, 2, 1, bias=False)
        self.dc3 = nn.ConvTranspose2d(ch // 4, ch // 8, 4, 2, 1, bias=False)
        self.dc4 = nn.ConvTranspose2d(ch // 8, 1, 3, 1, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(bottom_width * bottom_width * ch)
        self.bn1 = nn.BatchNorm2d(ch // 2)
        self.bn2 = nn.BatchNorm2d(ch // 4)
        self.bn3 = nn.BatchNorm2d(ch // 8)

    def __call__(self, z):
        h = F.relu(self.bn0(self.l0(z)))
        h = h.view(-1, self.ch, self.bottom_width, self.bottom_width)
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.tanh(self.dc4(h))
        return h


class Discriminator(nn.Module):
    def __init__(self, bottom_width=4, ch=512, wscale=0.02):
        super(Discriminator, self).__init__()
        self.bottom_width = bottom_width
        self.ch = ch

        self.c0_0 = nn.Conv2d(1, ch // 8, 3, 1, 1, bias=False)
        self.c0_1 = nn.Conv2d(ch // 8, ch // 4, 4, 2, 1, bias=False)
        self.c1_0 = nn.Conv2d(ch // 4, ch // 4, 3, 1, 1, bias=False)
        self.c1_1 = nn.Conv2d(ch // 4, ch // 2, 4, 2, 1, bias=False)
        self.c2_0 = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1, bias=False)
        self.c2_1 = nn.Conv2d(ch // 2, ch // 1, 4, 2, 1, bias=False)
        self.c3_0 = nn.Conv2d(ch // 1, ch // 1, 3, 1, 1, bias=False)
        self.l4 = nn.Linear(bottom_width * bottom_width * ch, 1)
        self.bn0_1 = nn.BatchNorm2d(ch // 4)
        self.bn1_0 = nn.BatchNorm2d(ch // 4)
        self.bn1_1 = nn.BatchNorm2d(ch // 2)
        self.bn2_0 = nn.BatchNorm2d(ch // 2)
        self.bn2_1 = nn.BatchNorm2d(ch // 1)
        self.bn3_0 = nn.BatchNorm2d(ch // 1)

    def __call__(self, x):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.bn0_1(self.c0_1(h)))
        h = F.leaky_relu(self.bn1_0(self.c1_0(h)))
        h = F.leaky_relu(self.bn1_1(self.c1_1(h)))
        h = F.leaky_relu(self.bn2_0(self.c2_0(h)))
        h = F.leaky_relu(self.bn2_1(self.c2_1(h)))
        h = F.leaky_relu(self.bn3_0(self.c3_0(h)))
        h = h.view(-1, self.ch * self.bottom_width * self.bottom_width)
        h = F.sigmoid(self.l4(h))
        return h
