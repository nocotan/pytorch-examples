# -*- coding: utf-8 -*-
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(in_dim, n_class)

    def forward(self, x):
        h = self.fc(x)
        return h
