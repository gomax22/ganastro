import torch.nn as nn
from torch.optim import Adam
from model.model import Generator, DiscriminatorDropout
from model.loss import inpainting_loss
from data.loader import NpzDataLoader
import torch
from torchvision.utils import save_image