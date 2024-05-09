import concurrent.futures
import os
from classes.observation import Observation
import numpy as np
import random
from tqdm import tqdm
from scipy import interpolate, special
import itertools
from typing import List
import concurrent

class Night:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    @classmethod
    def from_observations(cls, observations: List[Observation], wave_ref: np.ndarray | None = None, cutoff_begin: float | None = None, cutoff_end: float | None = None):
        return cls(wave_ref=wave_ref, cutoff_begin=cutoff_begin, cutoff_end=cutoff_end, observations=observations)
    
    @classmethod
    def from_directory(cls, night_path: str, cutoff_begin: float | None = None, cutoff_end: float | None = None):
        entries = sorted([os.path.join(night_path, entry) for entry in os.listdir(night_path) if entry.endswith("s1d_A.fits.gz")])
        observations: List[Observation] = [Observation.from_file(entry).preprocess() for entry in entries]
        return cls(cutoff_begin=cutoff_begin, cutoff_end=cutoff_end, observations=observations, entries=entries)
                    
                    
    def interpolate(self, progress=True, verbose=True):
        # check if all observations have the same number of pixels
        n_pixels = [obs.n_pixels for obs in self.observations]
        step_wl = [obs.step_wl for obs in self.observations]
        
        
        if ((max(n_pixels) == min(n_pixels)) and (max(step_wl) == min(step_wl))):
            if verbose:
                print("All observations have the same number of pixels and step wavelength.")
                print("No interpolation needed.")
            
        else:
            if verbose:
                print("Observations have different number of pixels or step wavelength.")
                print("Interpolating fluxes and errors to a common wavelength grid.")
            
            # observations have different wavelength ranges
            # interpolate fluxes and errors to a common wavelength grid
            start_wl = np.max([obs.start_wl for obs in self.observations])
            end_wl = np.min([obs.wave[-1] for obs in self.observations])
            n_pixels = np.mean([obs.n_pixels for obs in self.observations]).astype(int)
            self.wave_ref = np.linspace(start_wl, end_wl, n_pixels)
            
            if progress: progress_bar = tqdm(total=len(self.observations), desc="Interpolating...")
            
            for idx, obs in enumerate(self.observations):
                f_flux = interpolate.interp1d(obs.wave, obs.flux, kind="slinear")
                f_error = interpolate.interp1d(obs.wave, obs.error, kind="slinear")
                
                obs.wave = np.array(self.wave_ref, dtype=np.float32)
                obs.n_pixels = len(self.wave_ref)
                obs.start_wl = self.wave_ref[0]
                obs.flux = np.array(f_flux(self.wave_ref), dtype=np.float32)
                obs.error = np.array(f_error(self.wave_ref), dtype=np.float32)
                
                if progress: progress_bar.update(1)
                
        if progress: progress_bar.close()
        if verbose: print(f"Interpolation to a common wavelength grid completed.")
        return self
        
    def cutoff(self, verbose=True):
        if self.cutoff_begin is not None:
            idx_begin = np.argmin(np.abs(self.wave_ref - self.cutoff_begin))
        else:
            idx_begin = 0
        
        if self.cutoff_end is not None:
            idx_end = np.argmin(np.abs(self.wave_ref - self.cutoff_end))
        else:
            idx_end = len(self.wave_ref)
        
        for obs in self.observations:
            obs.wave = np.array(obs.wave[idx_begin:idx_end], dtype=np.float32)
            obs.n_pixels = len(obs.wave)
            obs.start_wl = obs.wave[0]
            obs.flux = np.array(obs.flux[idx_begin:idx_end], dtype=np.float32)
            obs.error = np.array(obs.error[idx_begin:idx_end], dtype=np.float32)
        if verbose: print(f"Cutoff of wavelength range completed.")
        return self
    
    def generate_night(self, combinations, out_path, date, idx):
        
        samples = []
        
        for c in combinations:    
            samples.append(Observation.from_observations([self.observations[idx] for idx in c], self.wave_ref))
        
        night = Night.from_observations(samples, wave_ref=self.wave_ref, cutoff_begin=self.cutoff_begin, cutoff_end=self.cutoff_end) \
            .interpolate(progress=False, verbose=False) \
            .cutoff(verbose=False)
            
        night.save(out_path, date, idx)
    
    
    def generate(self, k, samples_per_night, sampling_ratio, max_nights, out_path, date, concurrency=True, verbose=True):
        
        if verbose: print(f"Generating {samples_per_night} samples per night with {k} observations combined...")
        # generate combinations of n observations taken k at a time (n choose k)
        # generator offers great performance but has a sequential nature
        cb = list(itertools.combinations(range(len(self.observations)), k)) # we realize the generator 
        n_permutations = special.comb(len(self.observations), k)
        if verbose: print(f"Total number of possible permutations: {int(n_permutations)} (n choose k, n={len(self.observations)}, k={k})")
        
        # total number of possible nights
        n_nights = int(n_permutations / samples_per_night)
        if verbose: print(f"Total number of possible nights: {n_nights} (max_nights: {max_nights})")
        
        # limit the number of nights to generate
        if sampling_ratio != 1.0: 
            n_nights = int(n_nights * sampling_ratio)
            if verbose: print(f"Sampling ratio: {sampling_ratio} -> n_nights: {n_nights}")
        
        # limit the number of nights to generate up to max_nights
        if max_nights is not None and max_nights < n_nights:
            n_nights = max_nights
            
        limit = n_nights * samples_per_night
        if verbose: print(f"Limiting to {limit} samples. (nights: {n_nights}, samples per night: {samples_per_night})")
        
        # shuffle combinations
        random.shuffle(cb)
        cb = cb[:limit]
        
        if concurrency:
            pbar = tqdm(total=n_nights, desc="Generating nights...")
            args = [(cb[idx*samples_per_night:(idx+1)*samples_per_night], out_path, date, idx) for idx in range(n_nights)]
        
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = {executor.submit(self.generate_night, cbs, out_path, date, idx) for (cbs, out_path, date, idx) in args}
                
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
            pbar.close()
        else: 
            pbar = tqdm(total=n_nights, desc="Generating nights...")
            for idx in range(n_nights):
                self.generate_night(cb[idx*samples_per_night:(idx+1)*samples_per_night], out_path, date, idx)
                pbar.update(1)
            pbar.close()
        
        
    def save(self, out_path, date, idx):
        # initial additional parameters
        humidity = []
        pressure = []
        windspeed = []
        winddir = []
        temp10m = []
        airmass = []
        berv = []
        bervmx = []
        snr = []
        names = []
        flux = []
        wave = []
        error = []
        
        # extract additional parameters
        for obs in self.observations:
            humidity.append(obs.humidity)
            pressure.append(obs.pressure)
            windspeed.append(obs.windspeed)
            winddir.append(obs.winddir)
            temp10m.append(obs.temp10m)
            airmass.append(obs.airmass)
            berv.append(obs.berv)
            bervmx.append(obs.bervmx)
            snr.append(obs.snr)
            names.append(obs.name)
            flux.append(obs.flux)
            wave.append(obs.wave)
            error.append(obs.error)
        
        # save generated night
        np.savez_compressed(
            os.path.join(out_path, f"night_{date}_{idx}.npz"),
            flux=np.array(flux, dtype=np.float32), 
            error=np.array(error, dtype=np.float32), 
            wave=np.array(wave, dtype=np.float32),
            humidity=np.array(humidity, dtype=np.float32),
            pressure=np.array(pressure, dtype=np.float32),
            windspeed=np.array(windspeed, dtype=np.float32),
            winddir=np.array(winddir, dtype=np.float32),
            temp10m=np.array(temp10m, dtype=np.float32),
            airmass=np.array(airmass, dtype=np.float32),
            berv=np.array(berv, dtype=np.float32),
            bervmx=np.array(bervmx, dtype=np.float32),
            snr=np.array(snr, dtype=np.float32),
            names=np.array(names)
        )
        
        