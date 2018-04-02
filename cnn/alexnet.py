# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2),
        self.conv3 = nn.Conv2d(384, 256, kernel_size=3, padding=1),
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1),
        self.fc1 = nn.Linear(256 * 6 * 6, 4096),
        self.fc2 = nn.Linear(4096, 4096),
        self.fc3 = nn.Linear(4096, num_classes),

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.dropout(F.max_pool2d(h, 2))
        h = F.dropout(F.relu(self.fc1(h)))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)

        return h
