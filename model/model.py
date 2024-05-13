import torch
import torch.nn as nn
import numpy as np
from base import BaseModel

class GeneratorBlockDropout(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(GeneratorBlockDropout, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )
        self.apply(weights_init)
        
        
    def forward(self, x):
        return self.block(x)


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(GeneratorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.apply(weights_init)
        
        
    def forward(self, x):
        return self.block(x)


class Generator(BaseModel):
    def __init__(self, num_features, latent_dim, num_channels, n_layers=6):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.num_channels = num_channels
        self.n_layers = n_layers    

        self.net = nn.ModuleList()
        self.net.append(GeneratorBlock(latent_dim, num_features * (2 ** n_layers), 6, 1, 0)) # 100, 16 * (64), 6, 6
        
        layers = [2 ** i for i in range(n_layers, -1, -1)]
        for i in range(n_layers):
            self.net.append(GeneratorBlock(num_features * layers[i], num_features * layers[i+1], 4, 2, 1))
        
        self.net.append(nn.Sequential(
            nn.ConvTranspose2d(num_features, num_channels, 4, 2, 4, bias=False),
            nn.Tanh()
        ))
        
        self.apply(weights_init)
        

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class DiscriminatorBlockDropout(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(DiscriminatorBlockDropout, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )
        self.apply(weights_init)
        
        
    def forward(self, x):
        return self.block(x)

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(DiscriminatorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.apply(weights_init)
        
        
    def forward(self, x):
        return self.block(x)
    
    
class Discriminator(BaseModel):
    def __init__(self, num_features, num_channels, n_layers=6):
        super(Discriminator, self).__init__()
        self.num_features = num_features
        self.num_channels = num_channels
        self.n_layers = n_layers
        
        self.net = nn.ModuleList()
        self.net.append(nn.Sequential(
            nn.Conv2d(num_channels, num_features, 4, 2, 4, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        layers = [2 ** i for i in range(0, n_layers+1)]
        for i in range(n_layers):
            self.net.append(DiscriminatorBlock(num_features * layers[i], num_features * layers[i+1], 4, 2, 1))
        
        self.net.append(nn.Sequential(
            nn.Conv2d(num_features * (2 ** n_layers), 1, 6, 1, 0),
        ))
        
        self.apply(weights_init)


    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class GeneratorDropout(BaseModel):
    def __init__(self, num_features, latent_dim, num_channels, n_dropout=3, n_layers=6):
        super(GeneratorDropout, self).__init__()
        assert n_dropout <= n_layers
        
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.num_channels = num_channels
        self.n_layers = n_layers    

        self.net = nn.ModuleList()
        self.net.append(GeneratorBlockDropout(latent_dim, num_features * (2 ** n_layers), 6, 1, 0)) # 100, 16 * (64), 6, 6
        
        layers = [2 ** i for i in range(n_layers, -1, -1)]
        for i in range(n_dropout):
            self.net.append(GeneratorBlockDropout(num_features * layers[i], num_features * layers[i+1], 4, 2, 1))
            
            
        for i in range(n_dropout, n_layers+1):
            self.net.append(GeneratorBlockDropout(num_features * layers[i], num_features * layers[i+1], 4, 2, 1))
        
        self.net.append(nn.Sequential(
            nn.ConvTranspose2d(num_features, num_channels, 4, 2, 4, bias=False),
            nn.Tanh()
        ))
        
        self.apply(weights_init)
        

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

class DiscriminatorDropout(BaseModel):
    def __init__(self, num_features, num_channels, n_dropout=3, n_layers=6):
        super(DiscriminatorDropout, self).__init__()
        assert n_dropout <= n_layers
        
        self.num_features = num_features
        self.num_channels = num_channels
        self.n_layers = n_layers
        
        self.net = nn.ModuleList()
        self.net.append(nn.Sequential(
            nn.Conv2d(num_channels, num_features, 4, 2, 4, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        layers = [2 ** i for i in range(0, n_layers+1)]
        for i in range(n_dropout):
            self.net.append(DiscriminatorBlockDropout(num_features * layers[i], num_features * layers[i+1], 4, 2, 1))
            
        for i in range(n_dropout, n_layers+1):
            self.net.append(DiscriminatorBlockDropout(num_features * layers[i], num_features * layers[i+1], 4, 2, 1))
        
        self.net.append(nn.Sequential(
            nn.Conv2d(num_features * (2 ** n_layers), 1, 6, 1, 0),
        ))
        
        self.apply(weights_init)


    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
