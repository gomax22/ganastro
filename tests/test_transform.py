import numpy as np
from data.dataset import NpzDataset
from torchvision import transforms as T
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt
import random

DATADIR = '/projects/data/HARPN/K9_augmented2'


def test_transform():

    dataset = NpzDataset(DATADIR, (1, 762, 762))
    index = random.randint(0, len(dataset))
    flux, fname = dataset[index]
    
    data = np.load(fname)['flux']
    print(f"Filename: {fname}")
    print("\nOriginal: ")
    pprint({"data.shape": data.shape, 
            "data.dtype": data.dtype, 
            "data.min()": data.min(), 
            "data.max()": data.max(), 
            "data.mean()": data.mean(), 
            "data.std()": data.std()}
           )

    # value_ab = ((value_xy - x) / (y - x)) * (b - a) + a. -> [x, y]
    tdata = 2 * ((data - data.min())) / (data.max() - data.min()) - 1
    print("\nNormalized: ")
    pprint({"flux.shape": tdata.shape, 
            "flux.dtype": tdata.dtype, 
            "flux.min()": tdata.min(), 
            "flux.max()": tdata.max(), 
            "flux.mean()": tdata.mean(), 
            "flux.std()": tdata.std()}
           )
    
    print("\nChecking if the transformation is reversible: ")
    pprint({"flux.shape": flux.shape,
            "flux.dtype": flux.dtype,
            "flux.min()": flux.min(),
            "flux.max()": flux.max(),
            "flux.mean()": flux.mean(),
            "flux.std()": flux.std()})
    
    
    # value_xy = ((value_ab - a) / (b - a)) * (y - x) + x.
    sdata = (tdata + 1) / 2 * (data.max() - data.min()) + data.min()
    print("\nUnnormalized: ")
    pprint({"flux.shape": sdata.shape, 
            "flux.dtype": sdata.dtype, 
            "flux.min()": sdata.min(), 
            "flux.max()": sdata.max(), 
            "flux.mean()": sdata.mean(), 
            "flux.std()": sdata.std()}
           )
    
    data.resize(1, 762, 762)
    tdata.resize(1, 762, 762)
    sdata.resize(1, 762, 762)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    axs[0].imshow(data[0], cmap='gray')
    axs[0].set_title(f"index: {index}, {fname.split('/')[-1]}")
    axs[1].imshow(tdata[0], cmap='gray')
    axs[1].set_title("Normalized in [-1, 1]")
    axs[2].imshow(sdata[0], cmap='gray')
    axs[2].set_title(f"Unnormalized back to [{data.min():.2f}, {data.max():.2f}]")
    fig.savefig("transform_aug2.png", dpi=400)

if __name__ == "__main__":
    test_transform()