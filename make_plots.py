import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from model.model import Generator
from utils.util import prepare_device
from pathlib import Path
from argparse import ArgumentParser
from data.loader import NpzDataLoader

MODELS = [
    'gan_2/models/GANASTRO/0918_090751/checkpoint-epoch50.pth',
    'gan_2/models/GANASTRO/0918_231524/checkpoint-epoch100.pth',
    'gan_2/models/GANASTRO/0919_154305/checkpoint-epoch150.pth',
    'gan_2/models/GANASTRO/0920_033631/checkpoint-epoch200.pth',
    'gan_2/models/GANASTRO/0920_170346/checkpoint-epoch250.pth'
]


def make_training_loss_curves(**kwargs):
    csv_dir, output_dir = kwargs['csv_dir'], kwargs['output_dir']
    steps = []
    g_values = []
    d_values = []
    for entry in sorted(os.listdir(os.path.join(csv_dir, 'D_loss'))):
        df = pd.read_csv(os.path.join(csv_dir, 'D_loss', entry))
        d_values.extend(df['Value'].to_list())

    d_values = np.array(d_values)

    for entry in sorted(os.listdir(os.path.join(csv_dir, 'G_loss'))):
        df = pd.read_csv(os.path.join(csv_dir, 'G_loss', entry))
        steps.extend(df['Step'].to_list())
        g_values.extend(df['Value'].to_list())

    steps, g_values = np.array(steps), np.array(g_values)

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(exist_ok=True, parents=True)


    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, d_values, color='blue', lw=0.5, label='Discriminator')
    ax.plot(steps, g_values, color='orange', lw=0.5, label='Generator')
    ax.grid(True)
    # ax.set_yscale('log')
    ax.set_ylabel('Cross-entropy')
    ax.set_xlabel('Iterations')
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(output_dir, 'loss_curve_updated.png'), dpi=300)

def make_gif(**kwargs):
    output_dir = kwargs['output_dir']
    model = Generator(latent_dim=512, num_channels=1, num_features=8, n_layers=6)
    device, device_ids = prepare_device(4)
    model = model.to(device)

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    print('Generator loaded...')
    noise = torch.randn(1, 512, 1, 1).to(device)
    generated = model(noise).detach().cpu().numpy().squeeze()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(generated, cmap='gray', vmin=-1, vmax=1)
    ax.axis('off')
    fig.savefig(os.path.join(output_dir, 'fixed_noise_0.png'), dpi=300)

    for idx, model_path in enumerate(MODELS):
        model.load_state_dict(torch.load(model_path)['generator']['state_dict'])
        model.eval()

        with torch.no_grad():
            generated = model(noise).detach().cpu().numpy().squeeze()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(generated, cmap='gray', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')  
        fig.savefig(os.path.join(output_dir, f'fixed_noise_{idx}.png'), dpi=300)
        print(f'Generated image {idx}...')

def make_samples(**kwargs):
    output_dir = kwargs['output_dir']
    model = Generator(latent_dim=512, num_channels=1, num_features=8, n_layers=6)
    device, device_ids = prepare_device(4)
    model = model.to(device)

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.load_state_dict(torch.load(MODELS[-1])['generator']['state_dict'])
    model.eval()
    noise = torch.randn(12, 512, 1, 1).to(device)
    with torch.no_grad():
        generated = model(noise).detach().cpu().numpy().squeeze()

    for idx in range(noise.size(0)):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(generated[idx], cmap='gray', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')  
        fig.savefig(os.path.join(output_dir, f'samples_{idx}.png'), dpi=300)
        print(f'Generated image {idx}...')
    


def make_inpainting_loss(**kwargs):
    FOLDER = os.path.join(kwargs['folder'], kwargs['split'], kwargs['loss'])
    steps = []
    values_20 = []
    values_40 = []
    values_60 = []
    entries = sorted(os.listdir(FOLDER))
    
    # 20
    df = pd.read_csv(os.path.join(FOLDER, entries[0]))
    steps.extend(df['Step'].to_list())
    values_20.extend(df['Value'].to_list())

    # 40
    df = pd.read_csv(os.path.join(FOLDER, entries[1]))
    values_40.extend(df['Value'].to_list())

    # 60
    df = pd.read_csv(os.path.join(FOLDER, entries[2]))
    values_60.extend(df['Value'].to_list())

    steps, values_20, values_40, values_60 = np.array(steps), np.array(values_20), np.array(values_40), np.array(values_60)

    fig, ax = plt.subplots(figsize=(8, 5))
    # ax.plot(steps, values_40 + (np.random.randn(len(values_20)) / 1000), color='blue', lw=0.5, label='20% of masked pixels')
    ax.plot(steps, values_20, color='blue', lw=1.5, label='20% of masked pixels')
    ax.plot(steps, values_40, color='orange', lw=1.5, label='40% of masked pixels')
    ax.plot(steps, values_60, color='green', lw=1.5, label='60% of masked pixels')
    ax.grid(True)
    if kwargs['loss'] == 'inpaint':
        ax.set_ylabel(f'Inpainting Loss')
    elif kwargs['loss'] == 'context':
        ax.set_ylabel(f'Context Loss')
    elif kwargs['loss'] == 'prior':
        ax.set_ylabel(f'Perceptual Loss')
    ax.set_xlabel('Iterations')

    ax.legend(loc='upper right')
    fig.savefig(os.path.join(FOLDER, f"{kwargs['split']}_{kwargs['loss']}_loss_curve.png"),dpi=300)

def plot_real_nights(**kwargs):
    data_dir = kwargs['data_dir']
    output_dir = kwargs['output_dir']
    img_shape = kwargs['img_shape']
    num_spectra = kwargs['num_spectra']
    num_features = kwargs['num_features']

    loader = NpzDataLoader(data_dir, img_shape, batch_size=1, shuffle=False)

    fig, axs = plt.subplots(1, 6, figsize=(20, 4))
    for idx, (image, fname, min_max) in enumerate(loader):
        fname = fname[0].split(os.sep)[-1].split("_")[1]
        image = image.numpy()
        min_max_norm = min_max.squeeze().numpy()
        min_flux = min_max_norm[:num_spectra]
        max_flux = min_max_norm[num_spectra:]

        image = np.reshape(image, -1)[:-644] # cut last 644 pixels
        image = np.reshape(image, (num_spectra, num_features)) # reshape to original dimensions
        image = (image + 1.0) / 2.0 * (max_flux[..., np.newaxis] - min_flux[..., np.newaxis]) + min_flux[..., np.newaxis] # back to original domain (from -1, 1 to min_flux, max_flux)
        image.resize(img_shape)

        # from 20180708 to 2018-07-08
        fname = f"{fname[:4]}-{fname[4:6]}-{fname[6:]}"

        im = axs[idx].imshow(image.squeeze(), cmap='gray')
        axs[idx].set_title(fname)
        axs[idx].axis('off')
        axs[idx].patch.set_linewidth(5)
        axs[idx].patch.set_edgecolor('k')

    cax = axs[-1].inset_axes((1.05, 0, 0.08, 1.0))
    fig.colorbar(im, cax=cax)
    fig.tight_layout()

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    fig.savefig(os.path.join(output_dir, 'real_nights.png'), dpi=300)
    plt.close(fig)




if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--folder', default='inpaint-v3')
    ap.add_argument('--split', default='train')
    ap.add_argument('--loss', default='inpaint')
    ap.add_argument('--img_shape', default=(1, 762, 762))
    ap.add_argument('--num_spectra', default=58)
    ap.add_argument('--num_features', default=10000)
    ap.add_argument('--data_dir', default='/projects/data/HARPN/K9_preprocessed_v2')
    ap.add_argument('--output_dir', default='paper/images')
    ap.add_argument('--csv_dir', default='paper/csv')
    args = ap.parse_args()
    args = vars(args)
    print(args)
    

    func = locals().get(args['name'])
    func(**args)