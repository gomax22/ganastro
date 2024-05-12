import torch
import torch.nn as nn
from torchinfo import summary
import time
from model.model import Discriminator


class DiscriminatorSeq(nn.Module):
    def __init__(self, num_features, num_channels):
        super(DiscriminatorSeq, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 4, 2, 4, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            
            nn.Conv2d(num_features, num_features * (2 ** 1), 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * (2 ** 1)),
            nn.LeakyReLU(0.2, inplace=True),           
            
            
            nn.Conv2d(num_features * (2 ** 1), num_features * (2 ** 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * (2 ** 2)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features * (2 ** 2), num_features * (2 ** 3), 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * (2 ** 3)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features * (2 ** 3), num_features * (2 ** 4), 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * (2 ** 4)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features * (2 ** 4), num_features * (2 ** 5), 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * (2 ** 5)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features * (2 ** 5), num_features * (2 ** 6), 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * (2 ** 6)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features * (2 ** 6), 1, 6, 1, 0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    num_channels = 1
    num_features = 16
    start_time = time.time()
    discriminator = DiscriminatorSeq(num_features, num_channels)
    print("Time taken to instantiate DiscriminatorSeq: ", time.time() - start_time)
    print(discriminator)
    summary(discriminator, input_size=(1, 1, 762, 762))
    x = torch.randn(1, 1, 762, 762)
    
    start_time = time.time()
    print(discriminator(x).shape)
    print("Time taken for forward pass (cpu): ", time.time() - start_time)
    
    start_time = time.time()
    discriminator = Discriminator(num_features, num_channels, n_layers=6)
    print("Time taken to instantiate Discriminator: ", time.time() - start_time)
    print(discriminator)
    summary(discriminator, input_size=(1, 1, 762, 762))
    
    start_time = time.time()
    print(discriminator(x).shape)
    print("Time taken for forward pass (cpu): ", time.time() - start_time)
    