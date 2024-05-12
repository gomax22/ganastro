import torch
import torch.nn as nn
from torchinfo import summary
import time
from model.model import Generator


class GeneratorSeq(nn.Module):
    def __init__(self, latent_dim, num_channels, num_features):
        super(GeneratorSeq, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, num_features * (2 ** 6), 6, 1, 0, bias=False),
            nn.BatchNorm2d(num_features * (2 ** 6)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(num_features * (2 ** 6), num_features * (2 ** 5), 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * (2 ** 5)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(num_features * (2 ** 5), num_features * (2 ** 4), 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * (2 ** 4)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(num_features * (2 ** 4), num_features * (2 ** 3), 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * (2 ** 3)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(num_features * (2 ** 3), num_features * (2 ** 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * (2 ** 2)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(num_features * (2 ** 2), num_features * (2 ** 1), 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * (2 ** 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(num_features * (2 ** 1), num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(num_features, num_channels, 4, 2, 4, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    latent_dim = 100
    num_channels = 1
    num_features = 16
    start_time = time.time()
    generator = GeneratorSeq(latent_dim, num_channels, num_features)
    print("Time taken to instantiate GeneratorSeq: ", time.time() - start_time)
    print(generator)
    summary(generator, input_size=(1, latent_dim, 1, 1))
    x = torch.randn(1, latent_dim, 1, 1)
    
    start_time = time.time()
    print(generator(x).shape)
    print("Time taken for forward pass (cpu): ", time.time() - start_time)
    
    start_time = time.time()
    generator = Generator(num_features, latent_dim, num_channels, n_layers=6)
    print("Time taken to instantiate Generator: ", time.time() - start_time)
    print(generator)
    summary(generator, input_size=(1, latent_dim, 1, 1))
    
    start_time = time.time()
    print(generator(x).shape)
    print("Time taken for forward pass (cpu): ", time.time() - start_time)
    