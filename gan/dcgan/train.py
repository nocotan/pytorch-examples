# -*- coding: utf-8 -*-
import itertools
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from models import Generator, Discriminator


def main():
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.MNIST(root="../data",
                                     train=True,
                                     download=True,
                                     transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    gen = Generator()
    dis = Discriminator()

    criterion = nn.BCELoss()

    d_optimizer = torch.optim.Adam(dis.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0002)

    def train_discriminator(dis, x_real, y_real, x_fake, y_fake):
        dis.zero_grad()

        outputs = dis(x_real)
        real_loss = criterion(outputs, y_real)
        real_score = outputs

        outputs = dis(x_fake)
        fake_loss = criterion(outputs, y_fake)
        fake_score = outputs

        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        return d_loss, real_score, fake_score

    def train_generator(gen, dis_outputs, y_real):
        gen.zero_grad()
        g_loss = criterion(dis_outputs, y_real)
        g_loss.backward()
        g_optimizer.step()

        return g_loss

    test_noise = Variable(torch.randn(16, 100))
    size_figure_grid = int(math.sqrt(16))
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    num_epochs = 200
    num_batches = len(train_loader)
    num_fig = 0

    for epoch in range(num_epochs):
        for n, (images, _) in enumerate(train_loader):
            images = Variable(images)
            y_real = Variable(torch.ones(images.size(0)))

            z = Variable(torch.randn(images.size(0), 100))
            x_fake = gen(z)
            y_fake = Variable(torch.zeros(images.size(0),))

            d_loss, real_score, fake_score = train_discriminator(dis, images, y_real, x_fake, y_fake)

            z = Variable(torch.randn(images.size(0), 100))
            x_fake = gen(z)
            outputs = dis(x_fake)

            g_loss = train_generator(gen, outputs, y_real)

            if (n+1) % 10 == 0:
                test_images = gen(test_noise)
                for k in range(16):
                    i = k // 4
                    j = k % 4
                    ax[i, j].cla()
                    ax[i, j].imshow(test_images[k, :].cpu().data.numpy().reshape(32, 32), cmap="Greys")

                plt.savefig("results/res.png")
                num_fig += 1
                print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, '
                      'D(x): %.2f, D(G(z)): %.2f'
                      %(epoch + 1, num_epochs, n+1, num_batches, d_loss.data[0], g_loss.data[0],
                        real_score.data.mean(), fake_score.data.mean()))

        fig.close()


if __name__ == "__main__":
    main()
