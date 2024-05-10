import torch
import torch.nn as nn
import numpy as np


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(GeneratorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, num_features, latent_dim, num_channels, n_layers=6):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.num_channels = num_channels
        self.n_layers = n_layers    

        self.net = nn.ModuleList()
        self.net.append(GeneratorBlock(latent_dim, num_features * (2 ** n_layers), 6, 1, 0)) # 100, 16 * (64), 6, 6
        
        layers = [2 ** i for i in range(n_layers, -1, -1)]
        self.net.append([GeneratorBlock(num_features * layers[i], num_features * layers[i+1], 4, 2, 1) for i in range(n_layers)])
        self.net.append(nn.Sequential(
            nn.ConvTranspose2d(num_features, num_channels, 4, 2, 4, bias=False),
            nn.Tanh()
        ))
        

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


