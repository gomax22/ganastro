# Adapted from https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py

import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseInpainter
from utils.util import inf_loop, MetricTracker
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm

MASKING_LEVELS = {60: (11, 44), 40: (17, 39), 20: (22, 33)}

class Inpainter(BaseInpainter):
    """
    Inpainter class
    """
    def __init__(self, models, parameters, criterion, metric_ftns, optimizer, config, device, device_ids,
                 data_loader, valid_data_loader=None, 
                 lr_scheduler=None, len_epoch=None):
        super().__init__(models, parameters, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.device_ids = device_ids
        self.data_loader = data_loader

        self.masking_level = config.masking_level if hasattr(config, 'masking_level') else 20
        self.bounds = MASKING_LEVELS[self.masking_level]
        self.lamb = config.masking_level if hasattr(config, 'lamb') else 0.1
        self.wl_grid = config.wl_grid if hasattr(config, 'wl_grid') else [4240, 4340, 10000]
        self.shape = config.shape if hasattr(config, 'shape') else [58, 10000]

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_metrics = [MetricTracker('inpaint_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                                MetricTracker('context_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                                MetricTracker('perceptual_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                                MetricTracker('perceptual_loss_weighted', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                                MetricTracker('z_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer), 
                                MetricTracker('average_mse_inside_mask', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                                MetricTracker('psnr', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                                MetricTracker('ssim', *[m.__name__ for m in self.metric_ftns], writer=self.writer)]

        self.valid_metrics = [MetricTracker('inpaint_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                                MetricTracker('context_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                                MetricTracker('perceptual_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                                MetricTracker('perceptual_loss_weighted', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                                MetricTracker('z_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer), 
                                MetricTracker('average_mse_inside_mask', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                                MetricTracker('psnr', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                                MetricTracker('ssim', *[m.__name__ for m in self.metric_ftns], writer=self.writer)]

        # predefined  mask (11-44) (17-39) (22-33)
        mask = np.ones((58, 10000), dtype=np.float32)
        mask[self.bounds[0]:self.bounds[1], :] = 0.0
        
        # with resize, we mask also additional parameters such as min-max normalization params
        mask.resize(1, 1, 762, 762) # format NCHW

        # do not mask normalization params
        mask[:, :, -1, -644:] = 1.0
        self.mask = torch.from_numpy(mask).to(device)
                
        # eval models
        for idx in range(len(self.models)):
            self.models[idx].eval()

        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        for idx in range(len(self.train_metrics)):
            self.train_metrics[idx].reset()
            
        for batch_idx, (img, _, _) in enumerate(self.data_loader):
            # if batch_idx == (len(self.data_loader) - 1):
            #     continue

            # stack mask on dim 0 to match batch size
            mask = torch.cat([self.mask] * img.size(0), dim=0).to(self.device)
            
            real_labels = torch.ones(img.size(0)).to(self.device)
            real_imgs = img.to(self.device)
            self.optimizer.zero_grad()
            
            # recover images
            generated_imgs = self.models[0](self.parameters)
            
            # compute pred from discriminator
            fake_preds = self.models[1](generated_imgs).view(-1)
            
            # compute loss
            inpaint_loss, c_loss, prior_loss, prior_loss_weighted = self.criterion(
                real_imgs, generated_imgs, mask, fake_preds, real_labels, lamb=self.lamb)
            
            inpaint_loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            
            gradsum = torch.sum(self.parameters.grad.detach().cpu() ** 2)
            if gradsum > 1e-5:
                self.train_metrics[4].update('z_grad', gradsum.item())
                    
            # clamp parameters to [-1, 1]
            self.parameters.data.clamp_(-1, 1)
            
            self.train_metrics[0].update('inpaint_loss', inpaint_loss.item())
            self.train_metrics[1].update('context_loss', c_loss.item())
            self.train_metrics[2].update('perceptual_loss', prior_loss.item())
            self.train_metrics[3].update('perceptual_loss_weighted', prior_loss_weighted.item())
            
            # rmse = torch.sqrt(torch.sum((real_imgs * (1 - self.mask) - generated_imgs * (1 - self.mask)) ** 2) / torch.sum(1 - self.mask))
            mse = torch.sum((real_imgs * torch.abs(1 - mask) - generated_imgs[:real_imgs.size(0)] * torch.abs(1 - mask)) ** 2) / torch.abs(1 - mask).sum()
            self.train_metrics[5].update('average_mse_inside_mask', mse.item())
            
            # compute psnr, ssim between inpainted image and real image
            psnr = PeakSignalNoiseRatio().to(self.device)(real_imgs * mask + generated_imgs[:real_imgs.size(0)] * (1 - mask), real_imgs)
            ssim = StructuralSimilarityIndexMeasure().to(self.device)(real_imgs * mask + generated_imgs[:real_imgs.size(0)] * (1 - mask), real_imgs)
            self.train_metrics[6].update('psnr', psnr.item())
            self.train_metrics[7].update('ssim', ssim.item())
            if batch_idx % self.log_step == 0:
                self.logger.debug('Inpaint Epoch: {} {} Inpaint_Loss: {:.6f} Context_Loss: {:.6f} Prior_Loss: {:.6f} Prior_Loss_Weighted: {:.6f} PSNR: {:.6f} SSIM: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    inpaint_loss.item(),
                    c_loss.item(),
                    prior_loss.item(),
                    prior_loss_weighted.item(),
                    psnr.item(),
                    ssim.item())
                )
        
        logs = [self.train_metrics[idx].result() for idx in range(len(self.train_metrics))] 
        
        if self.do_validation:
            val_logs = self._valid_epoch(epoch)
            
            for log_t, log_v in zip(logs, val_logs):
                log_t.update(**{'val_'+k : v for k, v in log_v.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return logs

    def _progress(self, batch_idx, mode='train'):
        base = '[{}/{} ({:.0f}%)]'
        if mode == 'train':
            if hasattr(self.data_loader, 'n_samples'):
                current = batch_idx * self.data_loader.batch_size
                total = self.data_loader.n_samples
            else:
                current = batch_idx
                total = self.len_epoch
            return base.format(current, total, 100.0 * current / total)
        elif mode == 'valid':
            if hasattr(self.valid_data_loader, 'n_samples'):
                current = batch_idx * self.valid_data_loader.batch_size
                total = self.valid_data_loader.n_samples
            else:
                current = batch_idx
                total = len(self.valid_data_loader)
            return base.format(current, total, 100.0 * current / total)
        else:
            raise ValueError('mode should be train or valid')
    
    def _valid_epoch(self, epoch):
        
        for idx in range(len(self.valid_metrics)):
            self.valid_metrics[idx].reset()
        
        with torch.no_grad():
            for batch_idx, (img, _, _) in enumerate(self.valid_data_loader):
            
                # stack mask on dim 0 to match batch size
                mask = torch.cat([self.mask] * img.size(0), dim=0).to(self.device)
                
                real_labels = torch.ones(img.size(0)).to(self.device)
                real_imgs = img.to(self.device)
                
                # recover images
                generated_imgs = self.models[0](self.parameters)
                
                # compute pred from discriminator
                fake_preds = self.models[1](generated_imgs).view(-1)
                
                # compute loss
                inpaint_loss, c_loss, prior_loss, prior_loss_weighted = self.criterion(
                    real_imgs, generated_imgs, mask, fake_preds, real_labels, lamb=self.lamb)
                
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                gradsum = torch.sum(self.parameters.grad.detach().cpu() ** 2)
                if gradsum > 1e-5:
                    self.valid_metrics[4].update('z_grad', gradsum.item())
                        
                # clamp parameters to [-1, 1]
                self.parameters.data.clamp_(-1, 1)
                
                self.valid_metrics[0].update('inpaint_loss', inpaint_loss.item())
                self.valid_metrics[1].update('context_loss', c_loss.item())
                self.valid_metrics[2].update('perceptual_loss', prior_loss.item())
                self.valid_metrics[3].update('perceptual_loss_weighted', prior_loss_weighted.item())
                
                mse = torch.sum((real_imgs * torch.abs(1 - mask) - generated_imgs[:real_imgs.size(0)] * torch.abs(1 - mask)) ** 2) / torch.abs(1 - mask).sum()
                self.valid_metrics[5].update('average_mse_inside_mask', mse.item())
                
                # compute psnr, ssim
                psnr = PeakSignalNoiseRatio().to(self.device)(real_imgs * mask + generated_imgs[:real_imgs.size(0)] * (1 - mask), real_imgs)
                ssim = StructuralSimilarityIndexMeasure().to(self.device)(real_imgs * mask + generated_imgs[:real_imgs.size(0)] * (1 - mask), real_imgs)
                self.valid_metrics[6].update('psnr', psnr.item())
                self.valid_metrics[7].update('ssim', ssim.item())
                if batch_idx % self.log_step == 0:
                    self.logger.debug('Inpaint Val Epoch: {} {} Inpaint_Loss: {:.6f} Context_Loss: {:.6f} Prior_Loss: {:.6f} Prior_Loss_Weighted: {:.6f} PSNR: {:.6f} SSIM: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx, mode='valid'),
                        inpaint_loss.item(),
                        c_loss.item(),
                        prior_loss.item(),
                        prior_loss_weighted.item(),
                        psnr.item(),
                        ssim.item())
                    )
                    self.writer.add_image('input', make_grid(img.cpu(), nrow=8, normalize=True))
                    self.writer.add_image('masked_input', make_grid((real_imgs * mask).detach().cpu(), nrow=8, normalize=True))
                    self.writer.add_image('generated', make_grid(generated_imgs[:real_imgs.size(0)].cpu(), nrow=8, normalize=True))
                    self.writer.add_image('inpainted', make_grid((real_imgs * mask + generated_imgs[:real_imgs.size(0)] * (1 - mask)).detach().cpu(), nrow=8, normalize=True))
                    self.writer.add_image('mask', make_grid(mask.cpu(), nrow=8, normalize=True))
        
        logs = [self.valid_metrics[idx].result() for idx in range(len(self.valid_metrics))] 
            
        return logs

    
    def evaluate(self):
        print('Evaluating inpainter...')
        xx = np.linspace(self.wl_grid[0], self.wl_grid[1], self.wl_grid[2])
        
        with torch.no_grad():
            
            pbar = tqdm(total=len(self.data_loader) * self.data_loader.batch_size, desc="Evaluating nights...")
            for batch_idx, (img, _, _) in enumerate(self.data_loader):
                
                # create folder for batch
                batch_path = os.path.join(self.checkpoint_dir, 'output', f'batch_{batch_idx}')
                if not Path(batch_path).exists():
                    Path(batch_path).mkdir(parents=True, exist_ok=True)
                
                real_imgs = img.to(self.device)
                generated_img = self.models[0](self.parameters)
    
                # cut last 644 pixels (normalization params...)
                generated_img  = torch.reshape(generated_img, (generated_img.shape[0], -1))
                generated_img = generated_img[:, :-644]     # specific to [1, 762, 762] shape
                generated_img = torch.reshape(generated_img, (generated_img.shape[0], self.shape[0], self.shape[1]))
                
                real_imgs = torch.reshape(real_imgs, (real_imgs.shape[0], -1))
                real_imgs = real_imgs[:, :-644]     # specific to [1, 762, 762] shape
                real_imgs = torch.reshape(real_imgs, (real_imgs.shape[0],  self.shape[0], self.shape[1]))
                
                # save npz
                np.savez_compressed(os.path.join(batch_path, f'batch_{batch_idx}.npz'), 
                                    real_imgs=np.array(real_imgs.detach().cpu().numpy(), dtype=np.float32),
                                    generated_imgs=np.array(generated_img.detach().cpu().numpy(), dtype=np.float32),
                                    parameters=np.array(self.parameters.detach().cpu().numpy(), dtype=np.float32),
                                    mask=np.array(torch.cat([self.mask] * img.size(0), dim=0).cpu().numpy(), dtype=np.float32))
                
                for obs_idx, (real_spec, inpainted_spec) in enumerate(zip(real_imgs, generated_img)):
                    
                    # create folder for batch
                    obs_path = os.path.join(batch_path, f'obs_{obs_idx}')
                    if not Path(obs_path).exists():
                        Path(obs_path).mkdir(parents=True, exist_ok=True)

                    for idx, (real, inpainted) in enumerate(zip(real_spec, inpainted_spec)):
                        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                        ax.plot(xx, inpainted.detach().cpu().numpy(), color='black', linewidth=0.5, label='Generated')
                        ax.plot(xx, real.detach().cpu().numpy(), color='red', linewidth=0.5, label='Original')
                        ax.set_xlabel('Wavelength')
                        ax.set_ylabel('Normalized Counts between [-1, 1]')
                        ax.grid()
                        ax.legend()
                        
                        out_fname = os.path.join(obs_path, f'generated_spectrum_{idx}.png')
                        fig.savefig(out_fname, dpi=300)
                        plt.close(fig)
                        
                    pbar.update(1)