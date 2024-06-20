# 1. for each order
# 2.    nonfinite suppression (= 1)
# 3.    negative numbers suppression (= 1)
# 4.    log transformation

# 5.    median subtraction over spectra
# 6.    mean subtraction over spectral channels
# 7.    mean subtraction over spectra

# 8.    for each number of observation
# 9.        compute principal components up to current number of observation    
# 10.       get reconstructions
# 11.       measure sum-of-squares error between input image and reconstruction
# 12.       compute energy of the current number of principal components
# 13.       get optimal number of principal components
# 14.       get reconstructions
# 15.       measure sum-of-squares error between input image and reconstruction

# 16.   compute energy of the optimal number of principal components
# 17.   show reconstructions    

import argparse
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import os


def check_args(args):

    pass


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
    
    return preprocessed_flux
    
    
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


def main(args):
    
    # check arguments
    preprocessed = args['preprocessed']
    energy_threshold = args['energy_threshold']
    cutoff_begin = args['cutoff_begin']
    cutoff_end = args['cutoff_end']
    output = args['output']
    dataset = args['dataset']
    check_args(args)
    
    
    # load dataset
    flux = np.load(dataset)['flux'] # 58, 10000
    
    # preprocess flux 
    preprocessed_flux = preprocess(flux)
    
    # perform PCA analysis
    n_comps, recons, energy = analysis(preprocessed_flux, energy_threshold)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PCA analysis of HARPN spectra')
    ap.add_argument('-d', '--dataset', required=True, 
                    help='path to dataset file')
    ap.add_argument('-o', '--output', required=False,
                    help='path to output file')
    ap.add_argument('-e', '--energy-threshold', required=False, default=0.7, type=float,
                    help='energy threshold for optimal number of principal components')
    ap.add_argument('-p', '--preprocessed', required=False, default=False, type=bool,
                    help='preprocessed data')
    ap.add_argument('--cutoff-begin', required=False, default=0, type=float,
                    help='beginning of the cutoff')
    ap.add_argument('--cutoff-end', required=False, default=0, type=float,
                    help='end of the cutoff')
    args = vars(ap.parse_args())  
    
    main(args)