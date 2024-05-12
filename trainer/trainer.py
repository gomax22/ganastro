# Adapted from https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py

import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils.util import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, models, criterion, metric_ftns, optimizers, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(models, criterion, metric_ftns, optimizers, config)
        self.config = config
        self.device = device
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
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = [MetricTracker('G_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('real_acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('fake_acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer)]
                    
        self.fixed_noise =  torch.randn(64, self.models[0].latent_dim, 1, 1, device=self.device) # format NCHW
        
        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        
        for idx in range(2):
            self.models[idx].train()
        
        for idx in range(4):
            self.train_metrics[idx].reset()
            
            
        for batch_idx, (img, _) in enumerate(self.data_loader):
            losses = [None, None]
            batch_size = img.size(0)
            
            # real images
            real_imgs = img.to(self.device) 
            real_labels = torch.ones(batch_size, device=self.device)
            
            # generated (fake) images
            noise = torch.randn(batch_size, self.models[0].latent_dim, 1, 1, device=self.device)  # format NCHW
            fake_images = self.models[0](noise)
            fake_labels = torch.zeros(batch_size, device=self.device) # fake label = 0
            flipped_fake_labels = real_labels # here, fake label = 1
            
            
            # train discriminator
            self.optimizers[1].zero_grad()

            real_preds = self.models[1](real_imgs).view(-1)
            real_loss = self.criterion(real_preds, real_labels)
            
            fake_preds = self.models[1](fake_images.detach()).view(-1)
            fake_loss = self.criterion(fake_preds, fake_labels)
            
            losses[1] = 0.5 * (real_loss + fake_loss)
            losses[1].backward()
            self.optimizers[1].step()
            
            
            # train generator
            self.optimizers[0].zero_grad()
            
            fake_preds = self.models[1](fake_images).view(-1)
            losses[0] = self.criterion(fake_preds, flipped_fake_labels)
            losses[0].backward()
            self.optimizers[0].step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics[0].update('G_loss', losses[0].item())
            self.train_metrics[1].update('D_loss', losses[1].item())
            
            pred_labels_real = torch.where(real_preds.detach() > 0., 1., 0.)
            pred_labels_fake = torch.where(fake_preds.detach() > 0., 1., 0.)
            self.train_metrics[2].update('real_acc', (pred_labels_real == real_labels).float().mean() * 100.)
            self.train_metrics[3].update('fake_acc', (pred_labels_fake == fake_labels).float().mean() * 100.)
            
            """
            for met in self.metric_ftns:
                self.train_metrics[2].update(met.__name__, met(real_preds, real_labels))
                self.train_metrics[3].update(met.__name__, met(fake_preds, fake_labels))
            """ 
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    losses[0].item(),
                    losses[1].item()))
                self.writer.add_image('input', make_grid(img.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        
        logs = [self.train_metrics[idx].result() for idx in range(4)] 

        if self.do_validation:
            # val_log = self._valid_epoch(epoch)
            # log.update(**{'val_'+k : v for k, v in val_log.items()})
            self._valid_epoch(epoch)
        
        """
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        """
        return logs

    #def _valid_epoch(self, epoch):
    def _valid_epoch(self):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        for idx in range(2):
            self.models[idx].eval()
            
        with torch.no_grad():
            fake_images = self.models[0](self.fixed_noise).detach().cpu()
            self.writer.add_image('val', make_grid(fake_images, padding=2, normalize=True))
            
            
        # add histogram of model parameters to the tensorboard
        for model in self.models:
            for name, p in model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        
        return

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)