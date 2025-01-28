import argparse
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import os

def preprocess(flux):
    # working on a copy
    preprocessed_flux = np.array(flux)
    
    preprocessed_flux[np.where(np.isfinite(preprocessed_flux) == 0)] = 1
    preprocessed_flux[np.where(flux <= 1)] = 1
    preprocessed_flux = np.array(np.log10(preprocessed_flux))
    
    # median subtraction over spectra
    medians = np.sort(preprocessed_flux, axis=1)[:, int((preprocessed_flux.shape[-1] + 1) / 2)]
    preprocessed_flux -= medians[:, None]
    
    # mean subtraction over spectral channels
    means = np.mean(preprocessed_flux, axis=0)
    preprocessed_flux -= means[None, :]
    
    return preprocessed_flux, medians, means

def postprocess(flux, medians, means):
    # working on a copy
    postprocessed_flux = np.array(flux)
    
    # mean addition over spectral channels
    postprocessed_flux += means[None, :]
    
    # median addition over spectra
    postprocessed_flux += medians[:, None]
    
    # convert to original space
    postprocessed_flux = np.array(10 ** postprocessed_flux)
    
    return postprocessed_flux

    
    
def analysis(flux, energy_threshold):
    pca_input = np.array(flux)
    
    # pca_means = np.mean(pca_input, axis=0) 
    # pca_input -= pca_means[None, :]
    
    # compute PCA 
    pca_estimator = PCA(n_components=energy_threshold)
    pca_estimator.fit(pca_input)

    # get optimal number of principal components
    energy = np.sum(pca_estimator.explained_variance_ratio_)
    n_comps = pca_estimator.components_.shape[0]
    print('Optimal number of principal components: ', n_comps)
    
    # get reconstruction
    recons = pca_estimator.inverse_transform(pca_estimator.transform(pca_input))
    print('Reconstruction error: ', np.sum((pca_input - recons) ** 2))
    print('Energy conservation: ', energy)
    
    return [recons, n_comps, energy]

