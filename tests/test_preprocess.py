import numpy as np
from data.loader import NpzDataLoader
from pprint import pprint
import matplotlib.pyplot as plt
from classes.night import Night
from astropy.io import fits

DATADIR = '/projects/data/HARPN/K9_preprocessed/'

def test_preprocess():

    loader = NpzDataLoader(DATADIR, [1, 762, 762], batch_size=1, shuffle=False)   
    
    for batch_idx, (flux, fname) in enumerate(loader):
        print(f"Filename: {fname}")
        print("\nOriginal: ")
        pprint({"flux.shape": flux.shape, 
                "flux.dtype": flux.dtype, 
                "flux.min()": flux.min(), 
                "flux.max()": flux.max(), 
                "flux.mean()": flux.mean(), 
                "flux.std()": flux.std()}
            )

        # get night
        night_path = fname[0].split('/')[-1].split('_')[1]
        night_path = f'/projects/data/HARPN/K9/{night_path}/'
        
        # get cutoffs
        cutoff_begin = float(fname[0].split('/')[-1].split('_')[2][2:].split('.')[0])
        cutoff_end = float(fname[0].split('/')[-1].split('_')[3][2:].split('.')[0])
        
        # get observations from night
        night = Night.from_directory(night_path, cutoff_begin, cutoff_end)
        
        # interpolate the night observations
        night.interpolate()

        # cutoff wavelength range
        night.cutoff()
        
        # select a limited amount of observations
        night.select(58, region='center')
        
        data = np.array([obs.flux for obs in night.observations])
        
        
        # write a fits file 
        hdul = fits.HDUList([fits.PrimaryHDU(data)])
        hdul.writeto(f"{fname[0].split('/')[-1].split('.')[0]}.fits", overwrite=True)

        
        
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.imshow(data, cmap='gray')
        fig.savefig(f"original_{batch_idx}.png", dpi=600)
        plt.close(fig)
        
        min_flux, max_flux = data.min(axis=1), data.max(axis=1)
        print(f"Min: {min_flux.shape}, Max: {max_flux.shape}")
        data.resize(1, 762, 762, refcheck=False)
        
        fig, ax = plt.subplots(1, 3,figsize=(15, 15))
        ax[0].imshow(data[0], cmap='gray')
        ax[0].set_title("Original")
        
        
        # replace first zero padded with mean and std
        data[0, -1, -644:-644+58] = min_flux
        data[0, -1, -644+58:-644+58+58] = max_flux
        data = 2 * (data - data.min()) / (data.max() - data.min()) - 1
        
        ax[1].imshow(data[0], cmap='gray')
        ax[1].set_title("After manual processing")
        
        ax[2].imshow(flux[0][0], cmap='gray')
        ax[2].set_title(fname[0] + " from loader")
        fig.savefig(f"processed_{batch_idx}.png")
        plt.close(fig)
        

if __name__ == "__main__":
    test_preprocess()