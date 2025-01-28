#  Adapted from https://github.com/victoresque/pytorch-template/blob/master/data_loader/data_loaders.py

from base import BaseDataLoader
from torchvision import transforms
from data import dataset

class NpzDataLoader(BaseDataLoader):
    def __init__(self, data_dir, img_shape, batch_size, spectrum_normalization=True, shuffle=True, validation_split=0.0, num_workers=1, mode='comparison'):
        # trsfm = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5,), (0.5,))
        # ])
        self.data_dir = data_dir
        self.dataset = dataset.NpzDataset(self.data_dir, tuple(img_shape), spectrum_normalization=spectrum_normalization, mode=mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
