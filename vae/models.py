# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2_1 = nn.Linear(400, 20)
        self.fc2_2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2_1(h), self.fc2_2(h)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.sigmoid(self.fc4(h))
        return h

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
