# Adapted from https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py

import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseInpainter
from utils.util import inf_loop, MetricTracker

class Inpainter(BaseInpainter):
    """
    Inpainter class
    """
    def __init__(self, models, parameters, criterion, metric_ftns, optimizer, config, device, device_ids,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(models, parameters, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.device_ids = device_ids
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = [MetricTracker('inpaint_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('context_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('perceptual_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('perceptual_loss_weighted', *[m.__name__ for m in self.metric_ftns], writer=self.writer)]

        
        # predefined  mask
        mask = np.ones((data_loader.batch_size, 58, 10000), dtype=np.float32)
        start = mask.shape[1] // 5
        end = 4 * start
        mask[:, start:end, :] = 0.0
        
        # with resize, we mask also additional parameters such as min-max normalization params
        mask.resize(data_loader.batch_size, 1, 762, 762) # format NCHW
        self.mask = torch.from_numpy(mask).to(self.device)
        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        for idx in range(len(self.models)):
            self.models[idx].eval()
        
        for idx in range(len(self.train_metrics)):
            self.train_metrics[idx].reset()
            
            
        for batch_idx, (img, _) in enumerate(self.data_loader):
            
            real_imgs = img.to(self.device)
            self.optimizer.zero_grad()
            
            # recover images
            generated_imgs = self.models[0](self.parameters)
            
            # compute pred from discriminator
            fake_preds = self.models[1](generated_imgs).view(-1)
            
            # compute loss
            inpaint_loss, c_loss, prior_loss, prior_loss_weighted = self.criterion(real_imgs, generated_imgs, self.mask, fake_preds, lamb=0.1)
            inpaint_loss.backward()
            self.optimizer.step()
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics[0].update('inpaint_loss', inpaint_loss.item())
            self.train_metrics[1].update('context_loss', c_loss.item())
            self.train_metrics[2].update('perceptual_loss', prior_loss.item())
            self.train_metrics[3].update('perceptual_loss_weighted', prior_loss_weighted.item())
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Inpaint Epoch: {} {} Inpaint_Loss: {:.6f} Context_Loss: {:.6f} Prior_Loss: {:.6f} Prior_Loss_Weighted: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    inpaint_loss.item(),
                    c_loss.item(),
                    prior_loss.item(),
                    prior_loss_weighted.item())
                )
                self.writer.add_image('input', make_grid(img.cpu(), nrow=8, normalize=True))
                self.writer.add_image('masked_input', make_grid((real_imgs * self.mask).detach().cpu(), nrow=8, normalize=True))
                self.writer.add_image('inpainted', make_grid(generated_imgs.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        
        logs = [self.train_metrics[idx].result() for idx in range(len(self.train_metrics))] 
        
        return logs

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
        