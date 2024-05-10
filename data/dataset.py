# https://github.com/songquanpeng/pytorch-template/blob/master/data/dataset.py
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


from utils.file import list_all_images, list_sub_folders, exist_cache, load_cache, save_cache, safe_filename


class DefaultDataset(Dataset):
    """ No label. """

    def __init__(self, root, transform=None):
        self.samples = list_all_images(root)
        self.samples.sort()
        self.transform = transform

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        return self.load_image(self.samples[index])

    def __len__(self):
        return len(self.samples)


class FolderDataset(Dataset):
    """ Deprecated, use torchvision.datasets.ImageFolder instead. """

    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        self.targets = []
        self.classes = list_sub_folders(root)

        for i, class_ in enumerate(self.classes):
            filenames = list_all_images(class_)
            class_samples = filenames
            self.targets.extend([i] * len(class_samples))
            self.samples.extend(class_samples)

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        return self.load_image(self.samples[index]), self.targets[index]

    def __len__(self):
        return len(self.samples)


class NpzDataset(Dataset):
    """
    Deprecated, reference purpose only now.
    Sometimes we need more information, not only image and its corresponding label.
    We can use a npz file to offer this information.
    This file should have the following keys:
    1. samples: the path array of images, should be relative path (image_root).
    2. labels: corresponding labels.

    Notice: please make sure the order is correct among those attributes.
    """

    def __init__(self, data_dir, img_shape, transform=None):
        self.samples = [os.path.join(data_dir, entry) for entry in os.listdir(data_dir) if entry.endswith('_cb4240_ce4340.npz')]
        self.img_shape = img_shape
    
    def __getitem__(self, index):
        flux = np.load(self.samples[index])['flux']
        
        """
        flux = data['flux']
        humidity = data['humidity']
        pressure = data['pressure']
        windspeed = data['windspeed']
        winddir = data['winddir']
        temp = data['temp10m']
        airmass = data['airmass']
        berv = data['berv']
        bervmx = data['bervmx']
        snr = data['snr']
        # normalization parameters
        """
        """
        
        flux = data['flux'].reshape(762, 762).unsqueeze(0) # 58, 10000 -> 1, 762, 762
        transform = transforms.Compose([
            transforms.Resize([self.img_height, self.img_width])
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        """
        flux = flux.resize(self.img_size)
        
        if self.transform is not None:
            flux = self.transform(flux)
        return flux

    def __len__(self):
        return len(self.samples)