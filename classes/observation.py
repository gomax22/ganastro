import numpy as np
from astropy.io import fits
import os

# to compute BJD from MJD
from astropy.time import Time
from astropy import units as u, coordinates as coord

"""
    * read params from header
    * get flux
    * get error flux
    * get wavelength

"""

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
    def from_observations(cls, observations, wave_ref):
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
        
        for obs in observations:
            humidity += obs.humidity
            pressure += obs.pressure
            windspeed += obs.windspeed
            winddir += obs.winddir
            temp10m += obs.temp10m
            airmass += obs.airmass
            berv += obs.berv
            bervmx += obs.bervmx
            snr += obs.snr
            flux += obs.flux
            error += obs.error
        
        humidity /= len(observations)
        pressure /= len(observations)
        windspeed /= len(observations)
        winddir /= len(observations)
        temp10m /= len(observations)
        airmass /= len(observations)
        berv /= len(observations)
        bervmx /= len(observations)
        snr /= len(observations)  
        flux /= len(observations)
        error /= len(observations)       
        
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
   
   
    def preprocess(self):
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
        
        return self
        
        
        
        
        
        
        
        
