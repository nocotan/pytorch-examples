# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2)
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)

        return h
