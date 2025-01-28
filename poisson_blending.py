#
# original code at https://github.com/parosky/poissonblending
#

import numpy as np
import scipy.sparse
import pyamg
from data import dataset
from data.loader import NpzDataLoader
from trainer.inpainter import MASKING_LEVELS
import os
import matplotlib.pyplot as plt
from pathlib import Path
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import torch
from pprint import pprint
import json


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.numpy().tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))



def denorm(x):
  x = x.to('cpu')
  x = x.detach().numpy().squeeze()
  x = np.transpose(x, (1, 2, 0)) # C x H x W -> H x W x C
  x = x*np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  x = x.clip(0, 1)
  return x


def blend(img_target, img_source, img_mask):
    """
    Args:
        img_target: corrupted image
        img_source: generated image
        img_mask: patch mask
        
        returns -> final blended image
    """

    print(f"img_target.shape: {img_target.shape}") # 1, 58, 10000
    print(f"img_source.shape: {img_source.shape}") # 1, 58, 10000
    print(f"img_mask.shape: {img_mask.shape}") # 1, 58, 10000

    print(f"img_target.dtype: {img_target.dtype}") # float32
    print(f"img_source.dtype: {img_source.dtype}") # float32
    print(f"img_mask.dtype: {img_mask.dtype}") # float32
    

    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(img_target.shape), format='lil')
    for y in range(img_target.shape[0]):
        for x in range(img_target.shape[1]):
            if img_mask[y, x].all() >0 :
                index = x+y*img_target.shape[1]
                A[index, index] = 4
                if index+1 < np.prod(img_target.shape):
                    A[index, index+1] = -1
                if index-1 >= 0:
                    A[index, index-1] = -1
                if index+img_target.shape[1] < np.prod(img_target.shape):
                    A[index, index+img_target.shape[1]] = -1
                if index-img_target.shape[1] >= 0:
                    A[index, index-img_target.shape[1]] = -1
    A = A.tocsr()
    
    # create poisson matrix for b
    P = pyamg.gallery.poisson((img_mask.shape[0], img_mask.shape[1]))

    # get subimages
    t = img_target.flatten()
    s = img_source.flatten()

    # create b
    b = P * s
    for y in range(img_target.shape[0]):
        for x in range(img_target.shape[1]):
            if not img_mask[y,x].all()>0:
                index = x+y*img_target.shape[1]
                b[index] = t[index]

    # solve Ax = b
    x = pyamg.solve(A,b,verb=False,tol=1e-10)

    # assign x to target image
    x = np.reshape(x, img_target.shape)
    # x[x>255] = 255 # np.info(np.float32).max
    # x[x<0] = 0 # np.info(np.float32).eps
    x = np.array(x, img_target.dtype)[np.newaxis, ...]

    return x


# BEST CONFIGURATION (TEST):
# MSE, 'SUM', L = 0.01 (7th) [index = 6]

# BEST CONFIGURATION (TRAIN):
# L1, 'SUM', L = 0.1 (1st) [index = 0]




if __name__ == "__main__":
    
    # get current directory
    current_dir = os.getcwd()

    dirs = [os.path.join(current_dir, d, 'models', 'GANASTRO') for d in os.listdir(current_dir) if os.path.isdir(d) and d.startswith('gann_4_night')]
    dirs = sorted([os.path.join(d, os.listdir(d)[6]) for d in dirs if os.path.exists(os.path.join(d, os.listdir(d)[6]))])
    
    # make a list of list grouping the directories at 3
    dirs = [dirs[i:i+3] for i in range(0, len(dirs), 3)]
    # print(f"dirs: {dirs}")

    # load dataset
    data_dir = '/projects/data/HARPN/K9_preprocessed_v2'
    dataloader = NpzDataLoader(data_dir, (1, 762, 762), batch_size=1, spectrum_normalization=True, shuffle=False, validation_split=0.0, num_workers=4)
    valid_data_loader = dataloader.split_validation()

    xx = np.linspace(4240, 4340, 10000)

    if not Path('blended').exists():
        Path('blended').mkdir(parents=True, exist_ok=True)

    for idx, ((img, fname, min_max_norm), paths) in enumerate(zip(dataloader, dirs)):
        print(f"img_shape: {img.shape}")
        # normalization parameters:
        min_max_norm = min_max_norm.squeeze().numpy()
        min_flux = min_max_norm[:58][np.newaxis, ...]
        max_flux = min_max_norm[58:][np.newaxis, ...]

        print(f"min_flux: {min_flux.shape}")
        print(f"max_flux: {max_flux.shape}")

        for m_idx, dir_path in enumerate(paths):
            print(f"Processing {fname} and {dir_path}")
            bounds = MASKING_LEVELS[(m_idx + 1) * 20]
            print(f"bounds: {bounds}")
            data = np.load(os.path.join(dir_path, 'output', 'batch_0', 'batch_0.npz'))
            real_img = data['real_imgs']
            generated_img = data['generated_imgs']

            mask = np.ones((1, 58, 10000), dtype=np.float32)
            mask[:, bounds[0]:bounds[1], :] = 0.0

            print(f"real_img.shape: {real_img.shape}, real_img.dtype: {real_img.dtype}, real_img.min: {real_img.min()}, real_img.max: {real_img.max()}")
            print(f"generated_img.shape: {generated_img.shape}, generated_img.dtype: {generated_img.dtype}, generated_img.min: {generated_img.min()}, generated_img.max: {generated_img.max()}")
            
            
            if not Path(os.path.join('blended', dir_path.split(os.sep)[4])).exists():
                Path(os.path.join('blended', dir_path.split(os.sep)[4])).mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # divider = make_axes_locatable(ax[2])
            # cax = divider.append_axes('right', size='5%', pad=0.10)
            real_img_copy = np.copy(real_img)
            real_img_copy.resize(1, 762, 762)
            generated_img_copy = np.copy(generated_img)
            gen_img = np.copy(generated_img)
            generated_img_copy.resize(1, 762, 762)
            mask_copy = np.copy(mask)
            mask_copy.resize(1, 762, 762)

            vmin = min(real_img.min(), generated_img.min())
            vmax = max(real_img.max(), generated_img.max())
            ax[0].imshow(real_img_copy[0], vmin=vmin, vmax=vmax, cmap='gray')
            ax[0].set_title('Original Image')

            ax[1].imshow(generated_img_copy[0], vmin=vmin, vmax=vmax, cmap='gray')
            ax[1].set_title('Generated Image')

            im = ax[2].imshow(mask_copy[0], vmin=vmin, vmax=vmax, cmap='gray')
            ax[2].set_title('Mask Image')

            # fig.colorbar(im, cax=cax, orientation='vertical')
            fig.colorbar(im, ax=ax.ravel().tolist())
            fig.savefig(os.path.join('blended', dir_path.split(os.sep)[4], 'original.png'), dpi=300)
            plt.close(fig)


            # [-1, 1] to [0, 1]
            generated_img = (generated_img + 1.) / 2.
            real_img = (real_img + 1.) / 2.

            generated_img = generated_img.clip(0, 1)
            real_img = real_img.clip(0, 1)


            
            print(("Start poisson blending..."))
            blended_img = blend(real_img[0], generated_img[0], 1. - mask[0])

            blended_img_copy = np.copy(blended_img)
            blended_img_copy.resize(1, 762, 762)


            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # divider = make_axes_locatable(ax[2])
            # cax = divider.append_axes('right', size='5%', pad=0.10)
            real_img_copy = np.copy(real_img)
            real_img_copy.resize(1, 762, 762)
            generated_img_copy = np.copy(generated_img)
            generated_img_copy.resize(1, 762, 762)

            ax[0].imshow(real_img_copy[0], vmin=0.0, vmax=1.0, cmap='gray')
            ax[0].set_title('Original Image')

            ax[1].imshow(generated_img_copy[0], vmin=0.0, vmax=1.0, cmap='gray')
            ax[1].set_title('Generated Image')

            im = ax[2].imshow(blended_img_copy[0], vmin=0.0, vmax=1.0, cmap='gray')
            ax[2].set_title('Blended Image')

            # fig.colorbar(im, cax=cax, orientation='vertical')
            fig.colorbar(im, ax=ax.ravel().tolist())
            fig.savefig(os.path.join('blended', dir_path.split(os.sep)[4], 'blended_01.png'), dpi=300)
            plt.close(fig)


            # [0, 1] -> [-1, 1]
            blended_img = (blended_img * 2.0 - 1.0).clip(-1, 1)
            generated_img = (generated_img * 2.0 - 1.0).clip(-1, 1)
            real_img = (real_img * 2.0 - 1.0).clip(-1, 1)
            
            np.savez_compressed(os.path.join('blended', dir_path.split(os.sep)[4], 'poisson_11.npz'),
                                real_img=real_img,
                                generated_img=generated_img,
                                blended_img=blended_img)

            print(f"saved {os.path.join('blended', dir_path.split(os.sep)[4], 'poisson_11.npz')}")
            
            for obs_idx, (real_spec, inpainted_spec, blended_spec) in enumerate(zip(real_img, generated_img, blended_img)):
                
                # create folder for batch
                obs_path = os.path.join('blended', dir_path.split(os.sep)[4], f'obs_{obs_idx}')
                if not Path(obs_path).exists():
                    Path(obs_path).mkdir(parents=True, exist_ok=True)

                for idx, (real, inpainted, blnd) in enumerate(zip(real_spec, inpainted_spec, blended_spec)):
                    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                    ax.plot(xx, inpainted, color='black', linewidth=0.2, alpha=0.8, label='Generated')
                    ax.plot(xx, blnd, color='darkgreen', linewidth=0.2, alpha=0.8, label='Blended')
                    ax.plot(xx, real, color='red', linewidth=0.2, alpha=0.8, label='Original')
                    ax.set_xlabel('Wavelength')
                    ax.set_ylabel('Normalized Flux between [-1, 1]')
                    # ax.set_title('Generated spectrum')
                    ax.grid()
                    ax.legend()
                    
                    out_fname = os.path.join(obs_path, f'generated_spectrum_{idx}.png')
                    fig.savefig(out_fname, dpi=300)
                    plt.close(fig)
            print(f"saved spectra in {os.path.join('blended', dir_path.split(os.sep)[4], f'obs_{obs_idx}')}")

                        
            blended_img_copy = np.copy(blended_img)
            blended_img_copy.resize(1, 762, 762)
            real_img_copy = np.copy(real_img)
            real_img_copy.resize(1, 762, 762)
            generated_img_copy = np.copy(generated_img)
            generated_img_copy.resize(1, 762, 762)

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # divider = make_axes_locatable(ax[2])
            # cax = divider.append_axes('right', size='5%', pad=0.10)

            ax[0].imshow(real_img_copy[0], vmin=-1.0, vmax=1.0, cmap='gray')
            ax[0].set_title('Original Image')

            ax[1].imshow(generated_img_copy[0], vmin=-1.0, vmax=1.0, cmap='gray')
            ax[1].set_title('Generated Image')

            im = ax[2].imshow(blended_img_copy[0], vmin=-1.0, vmax=1.0, cmap='gray')
            ax[2].set_title('Blended Image')

            # fig.colorbar(im, cax=cax, orientation='vertical')
            fig.colorbar(im, ax=ax.ravel().tolist())
            fig.savefig(os.path.join('blended', dir_path.split(os.sep)[4], 'blended_11.png'), dpi=300)
            plt.close(fig)


            blended_img_copy = np.copy(blended_img[:, bounds[0]:bounds[1], :])
            real_img_copy = np.copy(real_img[:, bounds[0]:bounds[1], :])
            generated_img_copy = np.copy(generated_img[:, bounds[0]:bounds[1], :])

            # compute MSE, MAE, RMSE, SSIM, PSNR on images [-1, 1])
            metrics = {
                "images": {
                    "generated": {
                        "mse": np.mean((real_img_copy - generated_img_copy) ** 2),
                        "mae": np.mean(np.abs(real_img_copy - generated_img_copy)),
                        "rmse": np.sqrt(np.mean((real_img_copy - generated_img_copy) ** 2)),
                        "ssim": StructuralSimilarityIndexMeasure()(torch.from_numpy(real_img_copy[np.newaxis, ...]), torch.from_numpy(generated_img_copy[np.newaxis, ...])).item(),
                        "psnr": PeakSignalNoiseRatio()(torch.from_numpy(real_img_copy[np.newaxis, ...]), torch.from_numpy(generated_img_copy[np.newaxis, ...])).item()
                    },
                    "blended": {
                        "mse": np.mean((real_img_copy - blended_img_copy) ** 2),
                        "mae": np.mean(np.abs(real_img_copy - blended_img_copy)),
                        "rmse": np.sqrt(np.mean((real_img_copy - blended_img_copy) ** 2)),
                        "ssim": StructuralSimilarityIndexMeasure()(torch.from_numpy(real_img_copy[np.newaxis, ...]), torch.from_numpy(blended_img_copy[np.newaxis, ...])).item(),
                        "psnr": PeakSignalNoiseRatio()(torch.from_numpy(real_img_copy[np.newaxis, ...]), torch.from_numpy(blended_img_copy[np.newaxis, ...])).item()
                    }
                },
                "spectra": {
                    "generated": {
                        "max_mse": np.max([np.mean((real - inpainted) ** 2) for real, inpainted in zip(real_img_copy[0], generated_img_copy[0])]),
                        "max_mae": np.max([np.mean(np.abs(real - inpainted)) for real, inpainted in zip(real_img_copy[0], generated_img_copy[0])]),
                        "max_rmse": np.max([np.sqrt(np.mean((real - inpainted) ** 2)) for real, inpainted in zip(real_img_copy[0], generated_img_copy[0])]),
                        "min_cosine": np.min([np.dot(real, inpainted) / (np.linalg.norm(real) * np.linalg.norm(inpainted)) for real, inpainted in zip(real_img_copy[0], generated_img_copy[0])])
                    },
                    "blended": {
                        "max_mse": np.max([np.mean((real - blnd) ** 2) for real, blnd in zip(real_img_copy[0], blended_img_copy[0])]),
                        "max_mae": np.max([np.mean(np.abs(real - blnd)) for real, blnd in zip(real_img_copy[0], blended_img_copy[0])]),
                        "max_rmse": np.max([np.sqrt(np.mean((real - blnd) ** 2)) for real, blnd in zip(real_img_copy[0], blended_img_copy[0])]),
                        "min_cosine": np.min([np.dot(real, blnd) / (np.linalg.norm(real) * np.linalg.norm(blnd)) for real, blnd in zip(real_img_copy[0], blended_img_copy[0])])
                    }
                }
            }

            # compyte MSE, MAE, RMSE, cosine on spectra
            pprint(metrics)
            with open(os.path.join('blended', dir_path.split(os.sep)[4], 'metrics.json'), 'w') as f:
                json.dump(metrics, f, default=default)

            blended_img_copy = np.copy(blended_img)
            real_img_copy = np.copy(real_img)
            generated_img_copy = np.copy(generated_img)
            # [-1, 1] to [min_flux, max_flux]
            blended_img_copy = (blended_img_copy + 1.0) / 2.0 * (max_flux[..., np.newaxis] - min_flux[..., np.newaxis]) + min_flux[..., np.newaxis]
            generated_img_copy = (generated_img_copy + 1.0) / 2.0 * (max_flux[..., np.newaxis] - min_flux[..., np.newaxis]) + min_flux[..., np.newaxis]
            real_img_copy = (real_img_copy + 1.0) / 2.0 * (max_flux[..., np.newaxis] - min_flux[..., np.newaxis]) + min_flux[..., np.newaxis]

            # save
            np.savez_compressed(os.path.join('blended', dir_path.split(os.sep)[4], 'poisson_original.npz'),
                                real_img=real_img_copy,
                                generated_img=generated_img_copy,
                                blended_img=blended_img_copy)
            print(f"saved {os.path.join('blended', dir_path.split(os.sep)[4], 'poisson_original.npz')}")
            

            print("############################")
