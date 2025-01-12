import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
      super(ResidualBlock, self).__init__()
      self.relu = nn.ReLU()
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
      self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

class GeneratorResidualBlock(ResidualBlock):
    def __init__(self, in_channels, upsampling=True):
        super(GeneratorResidualBlock, self).__init__(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest') if upsampling else None

    def forward(self, z):
        out_conv1 = self.conv1(self.relu(self.bn1(z)))
        if self.upsampling is not None:
            out_conv1 = self.upsampling(out_conv1)
        out_conv2 = self.conv2(self.relu(self.bn2(out_conv1)))
        return out_conv2

class DescriminatorResidualBlock(ResidualBlock):
    def __init__(self, in_channels, downsampling=True):
        super(DescriminatorResidualBlock, self).__init__(in_channels, in_channels)
        self.downsampling = nn.AvgPool2d(kernel_size=2, stride=2) if downsampling else None

    def forward(self, z):
        out_conv1 = self.relu(self.conv1(z))
        out_conv2 = self.relu(self.conv2(out_conv1))
        if self.downsampling is not None:
            out_conv2 = self.downsampling(out_conv2)
        return out_conv2

class FirstDescriminatorResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, stride=2)

    def forward(self, z):
        out_conv1 = self.relu(self.conv1(z))
        out_conv2 = self.relu(self.conv2(out_conv1))
        return out_conv2

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.input_size = input_size
        self.fc = nn.Linear(input_size, input_size * 7 * 7)
        self.residual_block1 = GeneratorResidualBlock(input_size)
        self.residual_block2 = GeneratorResidualBlock(input_size)
        self.residual_block3 = GeneratorResidualBlock(input_size, upsampling=False)
        self.final_conv = nn.Conv2d(input_size, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        out_fc = self.fc(z)
        out_fc = self.relu(out_fc)
        z = out_fc.view(-1, self.input_size, 7, 7)
        out_1 = self.residual_block1(z)
        out_2 = self.residual_block2(out_1)
        out_3 = self.residual_block3(out_2)

        out_final_conv = self.final_conv(out_3)
        out = self.tanh(out_final_conv)

        return out

class Descriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Descriminator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.residual_block1 = FirstDescriminatorResidualBlock(input_size, output_size)
        self.residual_block2 = DescriminatorResidualBlock(output_size)
        self.residual_block3 = DescriminatorResidualBlock(output_size, downsampling=False)
        self.residual_block4 = DescriminatorResidualBlock(output_size, downsampling=False)
        self.mean_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc =nn.Linear(output_size, 1)

    def forward(self, z):
        out_1 = self.residual_block1(z)
        out_2 = self.residual_block2(out_1)
        out_3 = self.residual_block3(out_2)
        out_4 = self.residual_block4(out_3)
        out_mean_pool = self.mean_pool(self.relu(out_4))
        out_mean_pool = out_mean_pool.view(-1, self.output_size)
        out_fc = self.fc(out_mean_pool)

        return out_fc

def prepared_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

learning_rate = 2e-4
batch_size = 64
latent_dim = 100
in_channel = 1
mu = 0
sigma = 1

generator = Generator(latent_dim).to(device)
descriminator = Descriminator(in_channel, latent_dim).to(device)

n_descriminator = 5
dataloader = prepared_data(batch_size)
iteration_per_epoch = len(dataloader)
n_epochs = 10000 // iteration_per_epoch

def calc_lr(epoch, i):
    return learning_rate - ((epoch * iteration_per_epoch + i) / 10000) * learning_rate


def train_model(generator, descriminator, n_epochs, n_descriminator, dataloader):
    for epoch in range(n_epochs):
        for i, (real_imgs, _) in tqdm(enumerate(dataloader), desc="Training Progress",
                                      unit=f"/{iteration_per_epoch} batches"):
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=calc_lr(epoch, i))
            optimizer_D = torch.optim.Adam(descriminator.parameters(), lr=calc_lr(epoch, i))
            real_imgs = real_imgs.to(device)

            for j in range(n_descriminator):
                z = torch.normal(mean=mu, std=sigma, size=(batch_size, latent_dim)).to(device)
                fake_imgs = generator(z).to(device)

                descriminator_loss = -torch.mean(descriminator(real_imgs)) + torch.mean(
                    descriminator(fake_imgs.to(device)))
                optimizer_D.zero_grad()
                descriminator_loss.backward()
                optimizer_D.step()

                for p in descriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            z = torch.normal(mean=mu, std=sigma, size=(batch_size, latent_dim)).to(device)
            fake_imgs = generator(z).to(device)
            generator_loss = -torch.mean(descriminator(fake_imgs))

            optimizer_G.zero_grad()
            generator_loss.backward()
            optimizer_G.step()

        print(
            f"Epoch [{epoch + 1}/{n_epochs}] - Descriminator Loss: {descriminator_loss.item():.4f}, Generator Loss: {generator_loss.item():.4f}")

train_model(generator, descriminator, n_epochs, n_descriminator, dataloader)