import torch
import torch.nn as nn

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(DiscriminatorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)
    
    
class Discriminator(nn.Module):
    def __init__(self, num_features, num_channels, n_layers=6):
        super(Discriminator, self).__init__()
        self.num_features = num_features
        self.num_channels = num_channels
        self.n_layers = n_layers
        
        self.net = nn.ModuleList()
        self.net.append(DiscriminatorBlock(num_channels, num_features, 4, 2, 3))
        
        layers = [2 ** i for i in range(0, n_layers+1)]
        self.net.append([DiscriminatorBlock(num_features * layers[i], num_features * layers[i+1], 4, 2, 1) for i in range(n_layers)])
        self.net.append(nn.Sequential(
            nn.Conv2d(num_features * (2 ** n_layers), 1, 6, 1, 0),
            nn.Sigmoid()
        ))


    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x
    