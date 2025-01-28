# https://github.com/songquanpeng/pytorch-template/blob/master/data/dataset.py
import os
import numpy as np
from torch.utils.data import Dataset


class NpzDataset(Dataset):
    def __init__(self, data_dir, img_shape=(1, 762, 762), transform=None, spectrum_normalization=True, mode='training'):
        if mode == 'training':
            self.samples = [os.path.join(data_dir, entry) for entry in os.listdir(data_dir) if entry.startswith('night_20180708') and entry.endswith('_cb4240_ce4340.npz')]
            self.samples = [os.path.join(data_dir, entry) for entry in os.listdir(data_dir) if entry.startswith('night_20180722') and entry.endswith('_cb4240_ce4340.npz')]
            self.samples = [os.path.join(data_dir, entry) for entry in os.listdir(data_dir) if entry.startswith('night_20180723') and entry.endswith('_cb4240_ce4340.npz')]
            self.samples = [os.path.join(data_dir, entry) for entry in os.listdir(data_dir) if entry.startswith('night_20180904') and entry.endswith('_cb4240_ce4340.npz')]
            self.samples = [os.path.join(data_dir, entry) for entry in os.listdir(data_dir) if entry.startswith('night_20210905') and entry.endswith('_cb4240_ce4340.npz')]
        elif mode == 'inpainting':
            self.samples = [os.path.join(data_dir, entry) for entry in os.listdir(data_dir) if entry.startswith('night_20180901') and entry.endswith('_cb4240_ce4340.npz')]
        elif mode == 'comparison':
            self.samples = [os.path.join(data_dir, entry) for entry in os.listdir(data_dir) if entry.endswith('_cb4240_ce4340.npz')]
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        self.img_shape = img_shape
        self.transform = transform
        self.spectrum_normalization = spectrum_normalization
    
    def __getitem__(self, index):
        flux = np.load(self.samples[index])['flux']
        
        if self.spectrum_normalization:

            min_flux, max_flux = flux.min(axis=1), flux.max(axis=1)
            flux = 2 * (flux - min_flux[:, None]) / (max_flux - min_flux)[:, None] - 1
            
            # encode in [-1, 1]
            min_max = np.concatenate((min_flux, max_flux), axis=None)
            norm = 2 * (min_max - min_max.min()) / (min_max.max() - min_max.min()) - 1
            
            flux.resize(self.img_shape)
            flux[:, -1, -644:-644+58+58] = norm

        else:
            flux = 2 * (flux - flux.min()) / (flux.max() - flux.min()) - 1
            flux.resize(self.img_shape)

            min_max = np.array([flux.min(axis=1), flux.max(axis=1)])

        if self.transform is not None:
            flux = self.transform(flux)
        return (flux, self.samples[index], min_max)

    def __len__(self):
        return len(self.samples)
    