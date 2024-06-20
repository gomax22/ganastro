
# load parameters computed by inpainter
# get inpainted images
# restore to original domain

# preprocess images
# compute pca

# compute rmse
# show reconstructions accordingly

import argparse
import torch
import numpy as np
import data.loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils.util import prepare_device
import torch.nn as nn
from pca import preprocess, analysis
import os
from pathlib import Path


def check_args(args):
    
    # check if the input directory exists
    if not os.path.exists(args['dataset']):
        raise ValueError(f"Dataset directory {args['dataset']} does not exist.")
    
    # check if checkpoint exists
    if not os.path.exists(args['checkpoint']):
        raise ValueError(f"Checkpoint file {args['checkpoint']} does not exist.")
    
    # check if energy threshold is between 0 and 1
    if args['energy_threshold'] < 0 or args['energy_threshold'] > 1:
        raise ValueError("Energy threshold must be between 0 and 1.")
    
    # check if shape is a tuple
    if not isinstance(args['shape'], tuple):
        raise ValueError("Shape must be a tuple.")
    
    # check if output directory exists
    if not os.path.exists(args['output']):
        Path(args['output']).mkdir(parents=True, exist_ok=True)
    
    return True

def main(args):
    
    checkpoint = args['checkpoint']
    dataset = args['dataset']
    output = args['output']
    energy_threshold = args['energy_threshold']
    device = args['device']
    shape = args['shape']
    
    # load parameters computed by inpainter
    checkpoint = torch.load(checkpoint)
    parameters = checkpoint['parameters']
    print("Parameters are loaded.")
    
    # setup data_loader instances
    loader = checkpoint['config']['data_loader']['type']
    module_args = dict(checkpoint['config']['data_loader']['args'])
    dl_shape = module_args['img_shape']
    
    # TODO: change dataset dir eventually
    data_loader = getattr(module_data, loader)(**module_args)
    print("Data loader is set up.")
    
    # load generator from config
    arch = checkpoint['generator']['type']
    module_args = dict(checkpoint['generator']['args'])
    generator = getattr(module_arch, arch)(**module_args)
    generator.load_state_dict(checkpoint['generator']['state_dict'])
    print("Generator is set up.")
    
    # load discriminator from config
    arch = checkpoint['discriminator']['type']
    module_args = dict(checkpoint['discriminator']['args'])
    discriminator = getattr(module_arch, arch)(**module_args)
    discriminator.load_state_dict(checkpoint['discriminator']['state_dict'])
    print("Discriminator is set up.")
    
    print(generator)
    print(discriminator)
    print("Model architecture is set up.")
    
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(checkpoint['config']['n_gpu'])
    print("Device is set up.")
    generator, discriminator = generator.to(device), discriminator.to(device)

    if len(device_ids) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=device_ids)
        discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, checkpoint['config']['loss'])
    
    # set mask
    mask = np.ones((data_loader.batch_size, shape[0], shape[1]), dtype=np.float32)
    start = mask.shape[1] // 5
    end = 4 * start
    mask[:, start:end, :] = 0.0

    # with resize, we mask also additional parameters such as min-max normalization params
    mask.resize(data_loader.batch_size, dl_shape[0], dl_shape[1], dl_shape[2]) # format NCHW
    mask = torch.from_numpy(mask).to(device)

    # set evaluation mode
    generator.eval()
    discriminator.eval()
    parameters.requires_grad = False
    
    # compute inpainted images
    with torch.no_grad():
        for batch_idx, (img, fnames) in enumerate(data_loader):
                
            real_imgs = img.to(device)
            
            # recover images
            generated_imgs = generator(parameters)
            
            # compute pred from discriminator
            fake_preds = discriminator(generated_imgs).view(-1)
            
            # compute loss
            inpaint_loss, c_loss, prior_loss, prior_loss_weighted = criterion(real_imgs, generated_imgs, mask, fake_preds, lamb=0.1)
            print(f"Batch {batch_idx} - Inpaint loss: {inpaint_loss}, Context loss: {c_loss}, Prior loss: {prior_loss}, Prior loss weighted: {prior_loss_weighted}")
            
            # restore to original domain
            min_flux = generated_imgs[:, -1, -644:-644+58]
            max_flux = generated_imgs[:, -1, -644+58:-644+58+58]
            generated_imgs = (generated_imgs + 1) / 2 * (max_flux - min_flux) + min_flux
            
            # back to original shape
            generated_imgs = generated_imgs.detach().cpu().numpy()
            generated_imgs = generated_imgs.reshape(generated_imgs.shape[0], -1)[:, :-644]
            generated_imgs = generated_imgs.reshape(generated_imgs.shape[0], shape[0], shape[1])
            
            # preprocess images
            gan_processed_imgs = preprocess(generated_imgs)
            input_imgs = preprocess(img.detach().cpu().numpy())
            
            # compute residual
            residual_imgs = input_imgs - gan_processed_imgs

            # compute rmse
            print(f'Batch {batch_idx} - SSD: {np.sum(residual_imgs.detach().cpu().numpy() ** 2)}')
            
            pca_residuals   =   []
            pca_recons      =   []
            pca_comps       =   []
            pca_energies    =   []
            
            for idx, image in enumerate(input_imgs):
                
                # perform pca analysis
                recons, n_comp, energy = analysis(image, energy_threshold)    
                        
                # compute residuals
                residual_img = image - recons # log space
                
                # save image
                pca_residuals.append(residual_img)
                pca_recons.append(recons)
                pca_comps.append(n_comp)
                pca_energies.append(energy)
                        
            pca_residuals = np.array(pca_residuals)
            pca_comps = np.array(pca_comps)
            pca_energies = np.array(pca_energies)
            pca_recons = np.array(pca_recons)
            
            # save images
            np.savez_compressed(f"{output}/pca_residuals_{batch_idx}.npz",
                                images=img.detach().cpu().numpy(),
                                preprocessed=input_imgs, 
                                residuals=pca_residuals,
                                reconstructions=pca_recons,
                                components=pca_comps,
                                energies=pca_energies,
                                names=fnames)
            
            np.savez_compressed(f"{output}/gan_residuals_{batch_idx}.npz", 
                                images=img.detach().cpu().numpy(),
                                preprocessed=gan_processed_imgs,
                                residuals=residual_imgs.detach().cpu().numpy(),
                                reconstructions=generated_imgs,
                                names=fnames)
                
                
    
    
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Comparison between GANASTRO and PCA on HARPN data.")
    ap.add_argument("--checkpoint", required=True, 
                    help="path to checkpoint file for loading parameters  \
                        learned during inpainting")
    ap.add_argument("--dataset", default=None,
                    help="path to HARPN night data directory")
    ap.add_argument("--output", required=True, 
                    help="path to output directory for saving results")
    ap.add_argument('--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    ap.add_argument('--energy-threshold', default=0.9, type=float,
                      help='energy threshold for PCA')
    ap.add_argument('--shape', default=(58, 10000), type=tuple,
                      help='shape of the images')
    args = vars(ap.parse_args())
    check_args(args)
    main(args)