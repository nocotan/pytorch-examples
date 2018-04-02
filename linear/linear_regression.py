# -*- coding: utf-8 -*-
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        h = self.fc(x)
        return h
