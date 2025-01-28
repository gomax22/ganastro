import numpy as np
from astropy.io import fits
import os

# to compute BJD from MJD
from astropy.time import Time
from astropy import units as u, coordinates as coord
from astropy.stats import sigma_clip

class Observation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    @classmethod
    def from_file(cls, path):
        parent_dir = os.path.dirname(path)
        name = path.split(os.sep)[-1]
        real = True
        
        with fits.open(path) as hdul:
            header = hdul[0].header
            humidity = header['HIERARCH TNG METEO HUMIDITY']
            pressure = header['HIERARCH TNG METEO PRESSURE']
            windspeed = header['HIERARCH TNG METEO WINDSPEED']
            winddir = header['HIERARCH TNG METEO WINDDIR']
            temp10m = header['HIERARCH TNG METEO TEMP10M']
            airmass = header['AIRMASS']
            berv = header['HIERARCH TNG DRS BERV']
            bervmx = header['HIERARCH TNG DRS BERVMX']
            snr = header['HIERARCH TNG DRS SPE EXT SN53']
            jd_obs = Observation.mjd_to_bjd(header['MJD-OBS'] + ((header['EXPTIME']/2.)/3600./60.), header['RA-DEG'], header['DEC-DEG']).jd 
            n_pixels = header['NAXIS1']
            step_wl = header['CDELT1']
            start_wl = header['CRVAL1']
            flux = np.array(hdul[0].data, dtype=np.float32)
            error = np.sqrt(np.abs(flux)) + header['HIERARCH TNG DRS CCD SIGDET'] / header['HIERARCH TNG DRS CCD CONAD']
            wave = np.arange(n_pixels) * step_wl + start_wl
        
        return cls(parent_dir=parent_dir,
                   name=name,
                   real=real,
                   n_pixels=n_pixels,
                   step_wl=step_wl,
                   start_wl=start_wl,
                   wave=wave,
                   flux=flux,
                   error=error,
                   humidity=humidity,
                   pressure=pressure,
                   windspeed=windspeed,
                   winddir=winddir,
                   temp10m=temp10m,
                   airmass=airmass,
                   berv=berv,
                   bervmx=bervmx,
                   snr=snr, 
                   jd_obs=jd_obs)
        
    
    @classmethod
    def from_observations(cls, observations, wave_ref, weights=None, noise=None):
        assert weights is None or len(observations) == weights.shape[1], "The number of weights must be equal to the number of observations"
        assert noise is None or len(observations) == noise.shape[0], "The number of noise must be equal to the number of observations"

        if weights is None:
            weights = np.ones(len(observations)) / len(observations)

        if noise is None:
            noise = np.zeros(len(observations))
        
        parent_dir = observations[0].parent_dir
        name = [obs.name for obs in observations]
        real = False
        
        n_pixels = observations[0].n_pixels
        step_wl = observations[0].step_wl
        start_wl = observations[0].start_wl
        
        flux = np.zeros(n_pixels, dtype=np.float32)
        error = np.zeros(n_pixels, dtype=np.float32)
        
        humidity = 0.0
        pressure = 0.0
        windspeed = 0.0
        winddir = 0.0
        temp10m = 0.0
        airmass = 0.0
        berv = 0.0
        bervmx = 0.0
        snr = 0.0
        
        # TODO: very inefficient implementation, manual convolution
        # put all the features inside a np.ndarray and perform a dot product with the weights
        # most efficient implementation would be a batched dot product, but using this OOP approach this is not possible
        # (B, 1, seq_len) @ (seq_len, n_features) -> (B, 1, n_features) and then stack results on second dimension
        for obs, w, n in zip(observations, weights.T, noise):
            humidity += (w * obs.humidity + n)
            pressure += (w * obs.pressure + n)
            windspeed += (w * obs.windspeed + n)
            winddir += (w * obs.winddir + n)
            temp10m += (w * obs.temp10m + n)
            airmass += (w * obs.airmass + n)
            berv += (w * obs.berv + n)
            bervmx += (w * obs.bervmx + n)
            snr += (w * obs.snr + n)
            flux += (w * obs.flux + n)
            error += (w * obs.error + n)
        
        return cls(parent_dir=parent_dir, 
                   name=name, 
                   real=real, 
                   n_pixels=n_pixels, 
                   step_wl=step_wl, 
                   start_wl=start_wl, 
                   wave=wave_ref, 
                   flux=flux, 
                   error=error, 
                   humidity=humidity, 
                   pressure=pressure, 
                   windspeed=windspeed, 
                   winddir=winddir, 
                   temp10m=temp10m, 
                   airmass=airmass, 
                   berv=berv, 
                   bervmx=bervmx, 
                   snr=snr)           
        
    @staticmethod
    def mjd_to_bjd(mjd, ra_deg, dec_deg):
        # d : MJD value
        # UPDATE: Read RA, DEC from the header, in deg units. RA, DEC of the telescope pointing.
        # RA,DEC to compute the BJD precisely

        """
        Adopted from Jens
        """
        #Convert MJD to BJD to account for light travel time. Adopted from Astropy manual.
        t = Time(mjd,format='mjd',scale='tdb',location=coord.EarthLocation.from_geodetic(0,0,0))
        #target = coord.SkyCoord(RA,DEC,unit=(u.hourangle, u.deg), frame='icrs')
        target = coord.SkyCoord(ra_deg, dec_deg,unit=(u.deg, u.deg), frame='icrs')
        ltt_bary = t.light_travel_time(target)
        return t.tdb + ltt_bary # = BJD
   
   
    def preprocess(self, sigma_clipping=True):
        """    
        * remove nans
        * remove negative fluxes
        * remove negative errors
        * shift at the same wavelength -> interpolate
        * get wlen, flux, error interpolated
        
        * cutoff wavelength range if needed
        """
        
        # remove nans
        nan_mask = np.isnan(self.flux)
        self.flux = self.flux[~nan_mask]
        # self.wave = self.wave[~nan_mask]
        self.error = self.error[~nan_mask]
        
        # remove negative fluxes
        neg_mask = self.flux < 0.0
        self.flux[neg_mask] = 0.0
        self.error[neg_mask] = 0.0
        
        # sigma-clipping
        if sigma_clipping:
            self.flux = sigma_clip(self.flux, sigma_lower=6.0, sigma_upper=1.0, maxiters=None, cenfunc='median', stdfunc='std')
        
        return self
        
        
        
        
        
        
        
        
