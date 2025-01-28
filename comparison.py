
import argparse
import torch
import numpy as np
import data.loader as module_data
import model.loss as module_loss
import model.model as module_arch
from utils.util import prepare_device
from pca import preprocess, analysis, postprocess
import os
from pathlib import Path
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from scipy import stats
from pprint import pprint
import pickle
from trainer.inpainter import MASKING_LEVELS
import matplotlib.pyplot as plt
import numpy as np




def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.numpy().tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

# define metrics for comparison
# RMSE, MSE, MAE, PSNR, SSIM
# % Define performance metrics for residuals, PCA residual as ground-truth.
def compute_metrics(original_img, pca_recon, gan_recon, pca_residual, gan_residual, bounds=None, num_bins=100, hist_range=None):
    if bounds is not None:
        pca_recon = pca_recon[bounds[0]:bounds[1]]
        gan_recon = gan_recon[bounds[0]:bounds[1]]
        pca_residual = pca_residual[bounds[0]:bounds[1]]
        gan_residual = gan_residual[bounds[0]:bounds[1]]
        original_img = original_img[bounds[0]:bounds[1]]
    assert original_img.shape == pca_recon.shape == gan_recon.shape, "Shapes of images must be the same."


    # compute histograms
    pca_counts, pca_bin_edges = np.histogram(pca_residual.flatten(), bins=num_bins, range=hist_range if hist_range is not None else (pca_residual.min(), pca_residual.max()))
    gan_counts, gan_bin_edges = np.histogram(gan_residual.flatten(), bins=num_bins, range=hist_range if hist_range is not None else (gan_residual.min(), gan_residual.max()))

    pca_norm_counts = pca_counts / np.sum(pca_counts)
    gan_norm_counts = gan_counts / np.sum(gan_counts)

    # compute ks_tests
    ks_tests = {}
    num_samples = [pca_residual.shape[1] * i for i in range(1, pca_residual.shape[0] + 1)]

    for num_sample in num_samples:
        # take a random subset of the residuals
        indices = np.random.randint(0, np.prod(pca_residual.shape), num_sample)
        pca_sample = pca_residual.flatten()[indices]
        gan_sample = gan_residual.flatten()[indices]
        ks_tests[num_sample] = stats.ks_2samp(pca_sample, gan_sample)
    

    # compute metrics
    metrics = {
        # %   * between residual images:
        # %       * image similarity (MSE, MAE, SSIM, PSNR, cosine).
        # %       * energy of images**
        "originals": {
            "pca": {
                "mse": np.mean((original_img - pca_recon) ** 2).item(),
                "rmse": np.sqrt(np.mean((original_img - pca_recon) ** 2)).item(),
                "mae": np.mean(np.abs(original_img - pca_recon)).item(),
                "psnr": PeakSignalNoiseRatio()(torch.from_numpy(original_img), torch.from_numpy(pca_recon)).item(),
                "ssim": StructuralSimilarityIndexMeasure()(torch.from_numpy(original_img).unsqueeze(0).unsqueeze(0), torch.from_numpy(pca_recon).unsqueeze(0).unsqueeze(0)).item(),
            },
            "gan": {
                "mse": np.mean((original_img - gan_recon) ** 2).item(),
                "rmse": np.sqrt(np.mean((original_img - gan_recon) ** 2)).item(),
                "mae": np.mean(np.abs(original_img - gan_recon)).item(),
                "psnr": PeakSignalNoiseRatio()(torch.from_numpy(original_img).unsqueeze(0).unsqueeze(0), torch.from_numpy(gan_recon).unsqueeze(0).unsqueeze(0)).item(),
                "ssim": StructuralSimilarityIndexMeasure()(torch.from_numpy(original_img).unsqueeze(0).unsqueeze(0), torch.from_numpy(gan_recon).unsqueeze(0).unsqueeze(0)).item(),
            }
        },
        "residuals": {
            "mse": np.mean((gan_residual - pca_residual) ** 2).item(),
            "rmse": np.sqrt(np.mean((gan_residual - pca_residual) ** 2)).item(),
            "mae": np.mean(np.abs(gan_residual - pca_residual)).item(),
            "psnr": PeakSignalNoiseRatio()(torch.from_numpy(gan_residual).unsqueeze(0).unsqueeze(0), torch.from_numpy(pca_residual).unsqueeze(0).unsqueeze(0)).item(),
            "ssim": StructuralSimilarityIndexMeasure()(torch.from_numpy(gan_residual).unsqueeze(0).unsqueeze(0), torch.from_numpy(pca_residual).unsqueeze(0).unsqueeze(0)).item(),
            "ssds": [np.sum(gan_residual ** 2), np.sum(pca_residual ** 2)]
        },
        # %   * between inpainted / reconstructed images: 
        # %       * reconstruction error**
        # %       * image similarity (MSE, MAE, SSIM, PSNR, cosine) 
        # %           * already computed for inpainting
        "reconstructions": {
            "mse": np.mean((gan_recon - pca_recon) ** 2).item(),
            "rmse": np.sqrt(np.mean((gan_recon - pca_recon) ** 2)).item(),
            "mae": np.mean(np.abs(gan_recon - pca_recon)).item(),
            "psnr": PeakSignalNoiseRatio()(torch.from_numpy(gan_recon).unsqueeze(0).unsqueeze(0), torch.from_numpy(pca_recon).unsqueeze(0).unsqueeze(0)).item(),
            "ssim": StructuralSimilarityIndexMeasure()(torch.from_numpy(gan_recon).unsqueeze(0).unsqueeze(0), torch.from_numpy(pca_recon).unsqueeze(0).unsqueeze(0)).item(),
        },
        # %   * between distribution (histogram) of residuals:
        # %       * ks_test, KL, 
        # %       * any metric comparing distributions
        # %       * residual entropy
        # %       * comparing moments of the distribution, confidence intervals
        # Indeed, the p-value is lower than our threshold of 0.05, so we reject the null hypothesis in favor of the default “two-sided” alternative: the data were not drawn from the same distribution.
        # When both samples are drawn from the same distribution, we expect the data to be consistent with the null hypothesis most of the time.
        # As expected, the p-value of 0.54 is not below our threshold of 0.05, so we cannot reject the null hypothesis.
        "distributions": {
            "ks_tests": ks_tests,
            "kl_div": stats.entropy(pca_norm_counts, qk=gan_norm_counts).astype(np.float32),
            "moments": {
                "means": [np.mean(gan_residual), np.mean(pca_residual)],
                "stds": [np.std(gan_residual, ddof=1), np.std(pca_residual, ddof=1)],
                "mean_error": np.mean((gan_residual.mean(0) - pca_residual.mean(0)) / pca_residual.std(0)),
                "variance_ratio": np.mean((gan_residual.var(0) / pca_residual.var(0)))
            }, 
            "entropies": [stats.entropy(gan_norm_counts), stats.entropy(pca_norm_counts)], 
        },
        "spectra_pcagan": {
            "mse": [np.mean((pca_spec - gan_spec) ** 2) for pca_spec, gan_spec in zip(pca_recon, gan_recon)],
            "rmse": [np.sqrt(np.mean((pca_spec - gan_spec) ** 2)) for pca_spec, gan_spec in zip(pca_recon, gan_recon)],
            "mae": [np.mean(np.abs(pca_spec - gan_spec)) for pca_spec, gan_spec in zip(pca_recon, gan_recon)],
            "cosine": [np.dot(pca_spec, gan_spec) / (np.linalg.norm(pca_spec) * np.linalg.norm(gan_spec)) for pca_spec, gan_spec in zip(pca_recon, gan_recon)],
            
            
            "max_mse": np.max([np.mean((pca_spec - gan_spec) ** 2) for pca_spec, gan_spec in zip(pca_recon, gan_recon)]),
            "max_rmse": np.max([np.sqrt(np.mean((pca_spec - gan_spec) ** 2)) for pca_spec, gan_spec in zip(pca_recon, gan_recon)]),
            "max_mae": np.max([np.mean(np.abs(pca_spec - gan_spec)) for pca_spec, gan_spec in zip(pca_recon, gan_recon)]),
            "min_cosine": np.min([np.dot(pca_spec, gan_spec) / (np.linalg.norm(pca_spec) * np.linalg.norm(gan_spec)) for pca_spec, gan_spec in zip(pca_recon, gan_recon)]),
            "median_mae": np.median([np.mean(np.abs(pca_spec - gan_spec)) for pca_spec, gan_spec in zip(pca_recon, gan_recon)]),
        },
        "spectra_realgan": {
            "mse": [np.mean((original_spec - gan_spec) ** 2) for original_spec, gan_spec in zip(original_img, gan_recon)],
            "rmse": [np.sqrt(np.mean((original_spec - gan_spec) ** 2)) for original_spec, gan_spec in zip(original_img, gan_recon)],
            "mae": [np.mean(np.abs(original_spec - gan_spec)) for original_spec, gan_spec in zip(original_img, gan_recon)],
            "cosine": [np.dot(original_spec, gan_spec) / (np.linalg.norm(original_spec) * np.linalg.norm(gan_spec)) for original_spec, gan_spec in zip(original_img, gan_recon)],
            
            "max_mse": np.max([np.mean((original_spec - gan_spec) ** 2) for original_spec, gan_spec in zip(original_img, gan_recon)]),
            "max_rmse": np.max([np.sqrt(np.mean((original_spec - gan_spec) ** 2)) for original_spec, gan_spec in zip(original_img, gan_recon)]),
            "max_mae": np.max([np.mean(np.abs(original_spec - gan_spec)) for original_spec, gan_spec in zip(original_img, gan_recon)]),
            "min_cosine": np.min([np.dot(original_spec, gan_spec) / (np.linalg.norm(original_spec) * np.linalg.norm(gan_spec)) for original_spec, gan_spec in zip(original_img, gan_recon)]),
            "median_mae": np.median([np.mean(np.abs(original_spec - gan_spec)) for original_spec, gan_spec in zip(original_img, gan_recon)]),

        },
        "spectra_realpca": {
            "mse": [np.mean((original_spec - pca_spec) ** 2) for original_spec, pca_spec in zip(original_img, pca_recon)],
            "rmse": [np.sqrt(np.mean((original_spec - pca_spec) ** 2)) for original_spec, pca_spec in zip(original_img, pca_recon)],
            "mae": [np.mean(np.abs(original_spec - pca_spec)) for original_spec, pca_spec in zip(original_img, pca_recon)],
            "cosine": [np.dot(original_spec, pca_spec) / (np.linalg.norm(original_spec) * np.linalg.norm(pca_spec)) for original_spec, pca_spec in zip(original_img, pca_recon)],

            "max_mse": np.max([np.mean((original_spec - pca_spec) ** 2) for original_spec, pca_spec in zip(original_img, pca_recon)]),
            "max_rmse": np.max([np.sqrt(np.mean((original_spec - pca_spec) ** 2)) for original_spec, pca_spec in zip(original_img, pca_recon)]),
            "max_mae": np.max([np.mean(np.abs(original_spec - pca_spec)) for original_spec, pca_spec in zip(original_img, pca_recon)]),
            "min_cosine": np.min([np.dot(original_spec, pca_spec) / (np.linalg.norm(original_spec) * np.linalg.norm(pca_spec)) for original_spec, pca_spec in zip(original_img, pca_recon)]),
            "median_mae": np.median([np.mean(np.abs(original_spec - pca_spec)) for original_spec, pca_spec in zip(original_img, pca_recon)]),
        },
        "residual_spectra_pcagan": {
            "mse": [np.mean((pca_res - gan_res) ** 2) for pca_res, gan_res in zip(pca_residual, gan_residual)],
            "rmse": [np.sqrt(np.mean((pca_res - gan_res) ** 2)) for pca_res, gan_res in zip(pca_residual, gan_residual)],
            "mae": [np.mean(np.abs(pca_res - gan_res)) for pca_res, gan_res in zip(pca_residual, gan_residual)],
            "cosine": [np.dot(pca_res, gan_res) / (np.linalg.norm(pca_res) * np.linalg.norm(gan_res)) for pca_res, gan_res in zip(pca_residual, gan_residual)],

            "max_mse": np.max([np.mean((pca_res - gan_res) ** 2) for pca_res, gan_res in zip(pca_residual, gan_residual)]),
            "max_rmse": np.max([np.sqrt(np.mean((pca_res - gan_res) ** 2)) for pca_res, gan_res in zip(pca_residual, gan_residual)]),
            "max_mae": np.max([np.mean(np.abs(pca_res - gan_res)) for pca_res, gan_res in zip(pca_residual, gan_residual)]),
            "min_cosine": np.min([np.dot(pca_res, gan_res) / (np.linalg.norm(pca_res) * np.linalg.norm(gan_res)) for pca_res, gan_res in zip(pca_residual, gan_residual)]),

            "median_mae": np.median([np.mean(np.abs(pca_res - gan_res)) for pca_res, gan_res in zip(pca_residual, gan_residual)]),
            "median_distance": np.median(torch.norm((torch.from_numpy(pca_residual) - torch.from_numpy(gan_residual)), dim=-1).numpy().astype(np.float32))
        },

    }

    return metrics

def plot_images(
        input_img, 
        generated_img,
        pca_recon, 
        gan_recon, 
        pca_residual, 
        gan_residual, 
        img_shape, 
        output_path):
    
    input_img_c = np.copy(input_img)
    generated_img_c = np.copy(generated_img)
    pca_recon_c = np.copy(pca_recon)
    gan_recon_c = np.copy(gan_recon)
    pca_residual_c = np.copy(pca_residual)
    gan_residual_c = np.copy(gan_residual)

    input_img_c.resize(img_shape)
    generated_img_c.resize(img_shape)
    pca_recon_c.resize(img_shape)
    gan_recon_c.resize(img_shape)
    pca_residual_c.resize(img_shape)
    gan_residual_c.resize(img_shape)
    
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    vmin = min(input_img_c.min(), pca_recon_c.min(), gan_recon_c.min(), generated_img_c.min())
    vmax = max(input_img_c.max(), pca_recon_c.max(), gan_recon_c.max(), generated_img_c.max())
    ax[0, 0].imshow(input_img_c[0], vmin=vmin, vmax=vmax, cmap='gray')
    ax[0, 0].set_title('Original image', fontsize=12)

    ax[0, 1].imshow(generated_img_c[0], vmin=vmin, vmax=vmax, cmap='gray')
    ax[0, 1].set_title('Generated image', fontsize=12)

    ax[1, 0].imshow(pca_recon_c[0], vmin=vmin, vmax=vmax, cmap='gray')
    ax[1, 0].set_title('PCA Reconstruction', fontsize=12)

    im = ax[1, 1].imshow(gan_recon_c[0], vmin=vmin, vmax=vmax, cmap='gray')
    ax[1, 1].set_title('GAN Inpainted image', fontsize=12)

    fig.colorbar(im, ax=ax.ravel().tolist())
    fig.savefig(os.path.join(output_path, 'recons.png'), dpi=300)
    plt.close(fig)


    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    vmin = min(pca_residual_c.min(), gan_residual_c.min())
    vmax = max(pca_residual_c.max(), gan_residual_c.max())

    ax[0].imshow(pca_residual_c[0], vmin=vmin, vmax=vmax, cmap='viridis')
    ax[0].set_title('PCA Residual', fontsize=12)

    im = ax[1].imshow(gan_residual_c[0], vmin=vmin, vmax=vmax, cmap='viridis')
    ax[1].set_title('GAN Residual', fontsize=12)

    cax = ax[1].inset_axes((1.05, 0, 0.08, 1.0))
    fig.colorbar(im, cax=cax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, 'residuals.png'), dpi=300)

    plt.close(fig)
    print(f"saved images in {output_path}")


def plot_histograms(pca_residual, gan_residual, output_path, num_bins=100, hist_range=None, label='log'):    

    pca_counts, pca_bin_edges = np.histogram(pca_residual.flatten(), bins=num_bins, range=hist_range if hist_range is not None else (pca_residual.min(), pca_residual.max()))
    gan_counts, gan_bin_edges = np.histogram(gan_residual.flatten(), bins=num_bins, range=hist_range if hist_range is not None else (gan_residual.min(), gan_residual.max()))

    pca_norm_counts = pca_counts / np.sum(pca_counts)
    gan_norm_counts = gan_counts / np.sum(gan_counts)

    pca_ecdf = np.cumsum(pca_norm_counts)
    gan_ecdf = np.cumsum(gan_norm_counts)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.grid(alpha=0.5)
    ax.stairs(gan_norm_counts, gan_bin_edges, fill=True, color='red', alpha=0.6, label='GAN residual distribution')
    ax.stairs(pca_norm_counts, pca_bin_edges, fill=True, color='green', alpha=0.6, label='PCA residual distribution')
    ax2 = ax.twinx()
    
    ax2.plot(gan_bin_edges[1:], gan_ecdf, color='red', linestyle='dashed', alpha=0.6, label='GAN ECDF')
    ax2.plot(pca_bin_edges[1:], pca_ecdf, color='green', linestyle='dashed', alpha=0.6, label='PCA ECDF')
    ax.legend(fontsize=12, loc='upper left')
    ax.set_xlabel('Residual value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax2.set_ylabel('ECDF', fontsize=12)
    ax2.legend(fontsize=12, loc='lower right')
    fig.savefig(os.path.join(output_path, f'{label}-residual_histogram.png'), dpi=300)
    plt.close(fig)
    print(f"saved histograms in {output_path}")

def plot_spectra(input_imgs, pca_recons, gan_recons, pca_residual, gan_residual, obs_path):
    xx = np.linspace(4240, 4340, 10000)

    # create folder for batch
    if not Path(obs_path).exists():
        Path(obs_path).mkdir(parents=True, exist_ok=True)

    for idx, (real_spec, pca_spec, gan_spec, pca_res, gan_res) in enumerate(zip(input_imgs, pca_recons, gan_recons, pca_residual, gan_residual)):
        fig, ax = plt.subplots(2, 1, figsize=(8, 5), gridspec_kw={'height_ratios': [3, 1]})
        ax[0].plot(xx, real_spec, color='red', linewidth=0.5, alpha=0.8, label='Real')
        ax[0].plot(xx, gan_spec, color='black', linewidth=0.5, alpha=0.8, label='GAN')
        ax[0].plot(xx, pca_spec, color='darkgreen', linewidth=0.5, alpha=0.8, label='PCA')
        ax[0].set_ylabel('Counts', fontsize=12)
        ax[0].grid()
        ax[0].legend(fontsize=12)
        ax[1].scatter(xx, gan_res, s=0.5, edgecolors='black', facecolors='none', alpha=0.3, label='GAN Residual')
        ax[1].scatter(xx, pca_res, s=0.5, edgecolors='darkgreen', facecolors='none', alpha=0.3, label='PCA Residual')
        ax[1].plot(xx, np.zeros_like(xx), color='black', linestyle='dashed', alpha=0.8)
        ax[1].set_xlabel('Wavelength', fontsize=12)
        ax[1].set_ylabel('Residuals', fontsize=12)
        ax[1].grid()

        fig.tight_layout()
        out_fname = os.path.join(obs_path, f'generated_spectrum_{idx}.png')
        fig.savefig(out_fname, dpi=300)
        plt.close(fig)
    print(f"saved spectra in {obs_path}")

    for idx, (pca_res, gan_res) in enumerate(zip(pca_residual, gan_residual)):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(xx, pca_res, color='darkgreen', linewidth=0.5, alpha=0.8, label='PCA Residual')
        ax.plot(xx, gan_res, color='black', linewidth=0.5, alpha=0.8, label='GAN Residual')
        ax.set_xlabel('Wavelength', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.grid()
        ax.legend(fontsize=12)
        
        out_fname = os.path.join(obs_path, f'residual_spectrum_{idx}.png')
        fig.savefig(out_fname, dpi=300)
        plt.close(fig)


def run_comparison(args, checkpoint_path, dir_path, output_dir):
    print("Arguments:")
    pprint(args)
    
    dir_name = dir_path.split(os.sep)[-1]
    masking_level = dir_name.split("_")[-1]
    night_name = dir_name.split("_")[-2]

    energy_threshold = args['energy_threshold'] / 100
    device = args['device']
    shape = args['shape']
    wl_grid = args['wl_grid']
    
    # load parameters computed by inpainter
    checkpoint = torch.load(checkpoint_path)
    np_generated_imgs = np.load(os.path.join(output_dir, 'output', 'batch_0', 'batch_0.npz'))['generated_imgs']
    np_generated_imgs.resize(1, 1, 762, 762)
    print("Parameters are loaded.")

    config = checkpoint['config']
        
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    print("Data loader is set up.")
    module_args = dict(checkpoint['config']['data_loader']['args'])
    dl_shape = module_args['img_shape']
    
    # set mask
    mask = np.ones((data_loader.batch_size, shape[0], shape[1]), dtype=np.float32)
    bounds = MASKING_LEVELS[int(masking_level)]
    mask[:, bounds[0]:bounds[1], :] = 0.0

    # with resize, we mask also additional parameters such as min-max normalization params
    mask.resize(data_loader.batch_size, dl_shape[0], dl_shape[1], dl_shape[2]) # format NCHW

    # do not mask normalization params
    mask[:, :, -1, -644:] = 1.0
    mask = torch.from_numpy(mask).to(device)

    # iterate over data loader
    # DataLoader performs the following steps:
    # 1. load x
    # 2. preprocess x according to GAN
    with torch.no_grad():
        for batch_idx, (img, fnames, min_max) in enumerate(data_loader): # set shuffle = False on data loader / iterate with b = 1
            fname = fnames[0].split(os.sep)[-1].split("_")[1]

            if night_name != fname:
                print(f"Skipping {fname}...")
                continue
            real_imgs = img.to(device)
            
            # 3. build GAN reconstruction
            generated_imgs = torch.from_numpy(np_generated_imgs).to(device)

            print(f"assert real_imgs == generated_imgs: {torch.allclose(real_imgs.detach(), generated_imgs.detach())}")
            gan_recons = generated_imgs.detach() * (1.0 - mask) + real_imgs.detach() * mask
            gan_recons = gan_recons.squeeze().detach().cpu().numpy()
            real_imgs = real_imgs.squeeze().detach().cpu().numpy()
            generated_imgs = generated_imgs.squeeze().detach().cpu().numpy()
            
            output_path = os.path.join(output_dir, f"comparison_{args['energy_threshold']}", 'images')          
            if not os.path.exists(output_path):
                Path(output_path).mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(1, 3, figsize=(15, 8))

            vmin = min(real_imgs.min(), generated_imgs.min(), gan_recons.min())
            vmax = max(real_imgs.max(), generated_imgs.max(), gan_recons.max())
            ax[0].imshow(real_imgs, vmin=vmin, vmax=vmax, cmap='gray')
            ax[0].set_title('Original', fontsize=12)

            ax[1].imshow(generated_imgs, vmin=vmin, vmax=vmax, cmap='gray')
            ax[1].set_title('GAN Generated', fontsize=12)

            im = ax[2].imshow(gan_recons, vmin=vmin, vmax=vmax, cmap='gray')
            ax[2].set_title('GAN Reconstruction', fontsize=12)

            cax = ax[2].inset_axes((1.05, 0, 0.08, 1.0))
            fig.colorbar(im, cax=cax)
            fig.tight_layout()
            fig.savefig(os.path.join(output_path, 'normalized_recons.png'), dpi=300)
            plt.close(fig)
            print(f"saved images in {output_path}")
            
            # 4. extract normalization parameters
            # restore to original domain
            min_max_norm = min_max.squeeze().numpy()
            min_flux = min_max_norm[:shape[0]]
            max_flux = min_max_norm[shape[0]:]

            # 5. reshape GAN_recons to original dimensions and domain
            gan_recons = np.reshape(gan_recons, -1)[:-644]        # cut last 644 pixels
            gan_recons = np.reshape(gan_recons, (shape[0], shape[1]))  # reshape to original dimensiosn
    
            # 6. reshape real_imgs to original dimensions and domain
            real_imgs = np.reshape(real_imgs, -1)[:-644] # cut last 644 pixels
            real_imgs = np.reshape(real_imgs, (shape[0], shape[1])) # reshape to original dimensions
            
            generated_imgs = np.reshape(generated_imgs, -1)[:-644] # cut last 644 pixels
            generated_imgs = np.reshape(generated_imgs, (shape[0], shape[1])) # reshape to original dimensions

            output_path = os.path.join(output_dir, f"comparison_{args['energy_threshold']}", 'spectra')          
            if not os.path.exists(output_path):
                Path(output_path).mkdir(parents=True, exist_ok=True)

            xx = np.linspace(wl_grid[0], wl_grid[1], wl_grid[2])
            for idx, (real_spec, gan_spec) in enumerate(zip(real_imgs, gan_recons)):
                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                ax.plot(xx, real_spec, color='red', linewidth=0.5, alpha=0.8, label='Real')
                ax.plot(xx, gan_spec, color='black', linewidth=0.5, alpha=0.8, label='GAN')
                ax.set_xlabel('Wavelength', fontsize=12)
                ax.set_ylabel('Counts', fontsize=12)
                ax.grid()
                ax.legend(fontsize=12)
                
                out_fname = os.path.join(output_path, f'spectrum_{idx}.png')
                fig.savefig(out_fname, dpi=300)
                plt.close(fig)
            gan_recons = (gan_recons + 1.0) / 2.0 * (max_flux[..., np.newaxis] - min_flux[..., np.newaxis]) + min_flux[..., np.newaxis] # back to original domain (from -1, 1 to min_flux, max_flux)
            real_imgs = (real_imgs + 1.0) / 2.0 * (max_flux[..., np.newaxis] - min_flux[..., np.newaxis]) + min_flux[..., np.newaxis] # back to original domain (from -1, 1 to min_flux, max_flux)
            generated_imgs = (generated_imgs + 1.0) / 2.0 * (max_flux[..., np.newaxis] - min_flux[..., np.newaxis]) + min_flux[..., np.newaxis] # back to original domain (from -1, 1 to min_flux, max_flux)
            
            generated_img = np.copy(generated_imgs)
            generated_img.resize(1, 762, 762)

            real_img = np.copy(real_imgs)
            real_img.resize(1, 762, 762)

            gan_recon = np.copy(gan_recons)
            gan_recon.resize(1, 762, 762)
            
            output_path = os.path.join(output_dir, f"comparison_{args['energy_threshold']}", 'images')          
            if not os.path.exists(output_path):
                Path(output_path).mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(1, 3, figsize=(15, 8))

            vmin = min(real_img.min(), generated_img.min(), gan_recon.min())
            vmax = max(real_img.max(), generated_img.max(), gan_recon.max())
            ax[0].imshow(real_img[0], vmin=vmin, vmax=vmax, cmap='gray')
            ax[0].set_title('Original Image', fontsize=12)

            ax[1].imshow(generated_img[0], vmin=vmin, vmax=vmax, cmap='gray')
            ax[1].set_title('Generated Image', fontsize=12)

            im = ax[2].imshow(gan_recon[0], vmin=vmin, vmax=vmax, cmap='gray')
            ax[2].set_title('Inpainted Image', fontsize=12)

            cax = ax[2].inset_axes((1.05, 0, 0.08, 1.0))
            fig.colorbar(im, cax=cax)
            fig.tight_layout()
            fig.savefig(os.path.join(output_path, 'unnormalized_recons.png'), dpi=300)
            plt.close(fig)
            print(f"saved images in {output_path}")

            # 6.1 plot images in original domain
            output_path = os.path.join(output_dir, f"comparison_{args['energy_threshold']}", 'spectra')          
            if not os.path.exists(output_path):
                Path(output_path).mkdir(parents=True, exist_ok=True)
            for idx, (real_spec, gan_spec) in enumerate(zip(real_imgs, gan_recons)):
                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                ax.plot(xx, real_spec, color='red', linewidth=0.5, alpha=0.8, label='Real')
                ax.plot(xx, gan_spec, color='black', linewidth=0.5, alpha=0.8, label='GAN')
                ax.set_xlabel('Wavelength', fontsize=12)
                ax.set_ylabel('Counts', fontsize=12)
                ax.grid()
                ax.legend(fontsize=12)
                
                out_fname = os.path.join(output_path, f'original_spectrum_{idx}.png')
                fig.savefig(out_fname, dpi=300)
                plt.close(fig)

            # 7. preprocessing x_GAN, x according to PCA
            gan_recons, gan_medians, gan_means = preprocess(gan_recons)                     # log space
            pca_input_imgs, input_medians, input_means = preprocess(real_imgs)              # log space
            generated_imgs, gen_medians, gen_means = preprocess(generated_imgs)

            # 8. apply PCA
            pca_recons, pca_comps, pca_energies = analysis(pca_input_imgs, energy_threshold)    
            
            # 9. compute R_GAN and R_PCA residuals
            gan_residuals = pca_input_imgs - gan_recons                 # log space 
            pca_residuals = pca_input_imgs - pca_recons                 # log space
            
            # 9.1 back to original domain
            gan_recons = postprocess(gan_recons, gan_medians, gan_means)
            pca_input_imgs = postprocess(pca_input_imgs, input_medians, input_means)
            pca_recons = postprocess(pca_recons, input_medians, input_means)
            generated_imgs = postprocess(generated_imgs, gen_medians, gen_means)
                        
            # 9.2 plot images
            output_path = os.path.join(output_dir, f"comparison_{args['energy_threshold']}")          
            if not os.path.exists(output_path):
                Path(output_path).mkdir(parents=True, exist_ok=True)
            plot_images(pca_input_imgs, generated_imgs, pca_recons, gan_recons, pca_residuals, gan_residuals, dl_shape, output_path)
            
            # 9.3 plot spectra
            obs_path = os.path.join(output_path, "spectra", f"obs_{batch_idx}")
            plot_spectra(pca_input_imgs, pca_recons, gan_recons, pca_residuals, gan_residuals, obs_path) # in original domain as in log space they are not so meaningful

            # 9.4 plot histograms
            plot_histograms(pca_residuals, gan_residuals, output_path, hist_range=(-0.05, 0.05), label='log_-0.05_0.05')
            plot_histograms(pca_residuals, gan_residuals, output_path, label='log')

            gan_residual = np.copy(gan_residuals)
            pca_residual = np.copy(pca_residuals)
            gan_residual = 10 ** gan_residual
            pca_residual = 10 ** pca_residual
            plot_histograms(pca_residual, gan_residual, output_path, hist_range=(-0.05, 0.05), label='original_-0.05_0.05')
            plot_histograms(pca_residual, gan_residual, output_path, label='original')

            # 10. compute metrics for comparison

            # normalize between -1, 1
            vmin = min(pca_input_imgs.min(), pca_recons.min(), gan_recons.min())
            vmax = max(pca_input_imgs.max(), pca_recons.max(), gan_recons.max())
            pca_input_imgs = 2 * (pca_input_imgs - vmin) / (vmax - vmin) - 1
            pca_recons = 2 * (pca_recons - vmin) / (vmax - vmin) - 1
            gan_recons = 2 * (gan_recons - vmin) / (vmax - vmin) - 1

            metrics = compute_metrics(pca_input_imgs, pca_recons, gan_recons, pca_residuals, gan_residuals, bounds, hist_range=(-0.05, 0.05))
            print(f"\nMetrics for image {fnames[0]}")

            pprint(metrics)

            # 11. store results
            with open(os.path.join(output_path, 'metrics.pkl'), 'wb') as f:
                pickle.dump(metrics, f)
            print(f"saved metrics in {output_path}")
            

def main(args):

    # get current directory
    source_dir = args['source_dir']
    dir_paths = sorted([os.path.join(source_dir, d) for d in os.listdir(source_dir) if os.path.isdir(d) and d.startswith('gann_4_night')])
    dirs = [os.path.join(source_dir, d, 'models', 'GANASTRO') for d in os.listdir(source_dir) if os.path.isdir(d) and d.startswith('gann_4_night')]
    
    # expriments according to the best configuration (MSE, 'SUM', L = 0.01)
    experiments_dirs = sorted([os.path.join(d, os.listdir(d)[6]) for d in dirs if os.path.exists(os.path.join(d, os.listdir(d)[6]))])
    
    # make a list of list grouping the directories at 3
    for exp_dir, dir_path in zip(experiments_dirs, dir_paths):
        checkpoint_path = os.path.join(exp_dir, 'checkpoint-epoch20000.pth')
        print(f"\nRunning comparison for {checkpoint_path}...")
        print(f"Experiment directory: {exp_dir}")
        print(f"Directory path: {dir_path}")
        run_comparison(args, checkpoint_path, dir_path, exp_dir)

                
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Comparison between GANASTRO and PCA on HARPN data.")
    ap.add_argument('--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    ap.add_argument('--energy_threshold', default=90, type=float,
                      help='energy threshold for PCA (in percentage)')
    ap.add_argument('--shape', default=(58, 10000), type=tuple,
                      help='shape of the images')
    ap.add_argument('--wl_grid', default=(4240, 4340, 10000), type=tuple,
                        help='wavelength grid specifications')
    ap.add_argument('--source_dir', default='experiments/inpainting/', type=str,
                      help='source directory of the experiments')
    args = vars(ap.parse_args())
    main(args)
