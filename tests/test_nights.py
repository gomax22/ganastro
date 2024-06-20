import numpy as np
from astropy.io import fits
import os 

DATADIR = '/projects/data/HARPN/K9/'

def test_nights():

        entries = [os.path.join(DATADIR, entry) for entry in os.listdir(DATADIR) if os.path.isdir(os.path.join(DATADIR, entry)) and entry != "tars"]
        
        
        for entry in entries:
            samples = [os.path.join(DATADIR, entry, sample) for sample in os.listdir(entry) if sample.endswith('.fits.gz')]
            
            flux = []
            for sample in samples:
                with fits.open(sample) as hdul:
                    data = hdul[0].data
                    flux.append(data)
            
            # write a fits file 
            flux = np.array(flux, dtype=np.float32)
            hdul = fits.HDUList([fits.PrimaryHDU(flux)])
            hdul.writeto(f"{entry.split('/')[-1]}.fits", overwrite=True)

        
        

if __name__ == "__main__":
    test_nights()