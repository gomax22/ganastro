# Adapted from https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py

import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils.util import inf_loop, MetricTracker
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path
from torch.functional import F

OUTPUT_FOLDER = 'aae_output_aug'

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, models, criterion, metric_ftns, optimizers, config, device, device_ids,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(models, criterion, metric_ftns, optimizers, config)
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
        # self.do_validation = True
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = [MetricTracker('G_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_real', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_fake', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('real_acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('fake_acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('G_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer)]

        self.valid_metrics = [MetricTracker('G_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_real', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_fake', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('real_acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('fake_acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('G_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer)]

        
        latent_dim = self.config['generator']['args']['latent_dim']
        batch_size = self.config['data_loader']['args']['batch_size']
        self.fixed_noise =  torch.randn(batch_size, latent_dim, 1, 1, device=self.device) # format NCHW
        
        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        
        latent_dim = self.config['generator']['args']['latent_dim']
        
        for idx in range(len(self.models)):
            self.models[idx].train()
        
        for idx in range(len(self.train_metrics)):
            self.train_metrics[idx].reset()
            
            
        for batch_idx, (img, _, _) in enumerate(self.data_loader):
            losses = [None, None]
            batch_size = img.size(0)
            
            # real images
            real_imgs = img.to(self.device) 
            
            # one-sided label smoothing
            real_labels = torch.ones(batch_size, device=self.device) * 0.9
            
            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=self.device)  # format NCHW
            fake_images = self.models[0](noise)
            
            fake_labels = torch.zeros(batch_size, device=self.device) # fake label = 0
            # flipped_fake_labels = real_labels # here, fake label = 1
            
            
            # train discriminator
            self.optimizers[1].zero_grad()

            real_preds = self.models[1](real_imgs).view(-1)
            real_loss = self.criterion(real_preds, real_labels)
            
            fake_preds = self.models[1](fake_images.detach()).view(-1)
            fake_loss = self.criterion(fake_preds, fake_labels)
            
            losses[1] = 0.5 * (real_loss + fake_loss)
            losses[1].backward()
            self.optimizers[1].step()
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            # print discriminator gradients
            gradsum = 0.0
            for p in self.models[1].parameters():
                if p.grad is not None:
                    gradsum += torch.sum(p.grad.detach().cpu() ** 2)
            if gradsum > 1e-5:
                self.train_metrics[6].update('D_grad', gradsum.item())
            
            # train generator
            self.optimizers[0].zero_grad()
            
            fake_preds = self.models[1](fake_images).view(-1)
            losses[0] = self.criterion(fake_preds, real_labels)
            losses[0].backward()
            self.optimizers[0].step()

            # print generator gradients
            gradsum = 0.0
            for p in self.models[0].parameters():
                if p.grad is not None:
                    gradsum += torch.sum(p.grad.detach().cpu() ** 2)
            if gradsum > 1e-5:
                self.train_metrics[7].update('G_grad', gradsum.item())

            self.train_metrics[0].update('G_loss', losses[0].item())
            self.train_metrics[1].update('D_loss', losses[1].item())
            self.train_metrics[2].update('D_real', real_loss.item())
            self.train_metrics[3].update('D_fake', fake_loss.item())
            
            pred_labels_real = torch.where(F.sigmoid(real_preds.detach()) > 0.5, 1., 0.)
            pred_labels_fake = torch.where(F.sigmoid(fake_preds.detach()) > 0.5, 1., 0.)
            
            self.train_metrics[4].update('real_acc', (pred_labels_real == (real_labels + 0.1)).float().mean() * 100.)
            self.train_metrics[5].update('fake_acc', (pred_labels_fake == fake_labels).float().mean() * 100.)
            
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
                # self.writer.add_image('input', make_grid(img.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        
        logs = [self.train_metrics[idx].result() for idx in range(len(self.train_metrics))] 
        
        ## validation
        """
        for idx in range(len(self.models)):
            self.models[idx].eval()
            
        with torch.no_grad():
            fake_images = self.models[0](self.fixed_noise).detach().cpu()
            self.writer.add_image('fixed_noise', make_grid(fake_images, padding=2, normalize=True))
        """
        if self.do_validation:
            val_logs = self._valid_epoch(epoch)
            
            for log_t, log_v in zip(logs, val_logs):
                log_t.update(**{'val_'+k : v for k, v in log_v.items()})
        
        """
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        """
        return logs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        for idx in range(len(self.models)):
            self.models[idx].eval()
            
        for idx in range(len(self.valid_metrics)):
            self.valid_metrics[idx].reset()
        
        latent_dim = self.config['generator']['args']['latent_dim']
        
        with torch.no_grad():
            for batch_idx, (img, _, _) in enumerate(self.valid_data_loader):
                
                losses = [None, None]
                batch_size = img.size(0)
                
                # real images
                real_imgs = img.to(self.device) 
                real_labels = torch.ones(batch_size, device=self.device)
                
                # generated (fake) images
                noise = torch.randn(batch_size, latent_dim, 1, 1, device=self.device)  # format NCHW
                fake_images = self.models[0](noise)
                fake_labels = torch.zeros(batch_size, device=self.device) # fake label = 0
                
                real_preds = self.models[1](real_imgs).view(-1)
                real_loss = self.criterion(real_preds, real_labels)
                
                fake_preds = self.models[1](fake_images.detach()).view(-1)
                fake_loss = self.criterion(fake_preds, fake_labels)
                
                losses[1] = 0.5 * (real_loss + fake_loss)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                gradsum = 0.0
                for p in self.models[1].parameters():
                    if p.grad is not None:
                        gradsum += torch.sum(p.grad.detach().cpu() ** 2)
                if gradsum > 1e-5:
                    self.valid_metrics[6].update('D_grad', gradsum.item())

                # validate generator
                fake_preds = self.models[1](fake_images).view(-1)
                losses[0] = self.criterion(fake_preds, fake_labels)

                gradsum = 0.0
                for p in self.models[0].parameters():
                    if p.grad is not None:
                        gradsum += torch.sum(p.grad.detach().cpu() ** 2)
                if gradsum > 1e-5:
                    self.valid_metrics[7].update('G_grad', gradsum.item())


                self.valid_metrics[0].update('G_loss', losses[0].item())
                self.valid_metrics[1].update('D_loss', losses[1].item())
                self.valid_metrics[2].update('D_real', real_loss.item())
                self.valid_metrics[3].update('D_fake', fake_loss.item())
                
                pred_labels_real = torch.where(F.sigmoid(real_preds.detach()) > 0.5, 1., 0.)
                pred_labels_fake = torch.where(F.sigmoid(fake_preds.detach()) > 0.5, 1., 0.)
                self.valid_metrics[4].update('real_acc', (pred_labels_real == real_labels).float().mean() * 100.)
                self.valid_metrics[5].update('fake_acc', (pred_labels_fake == fake_labels).float().mean() * 100.)
                
                if batch_idx % self.log_step == 0:
                    self.logger.debug('Val Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx, mode='valid'),
                        losses[0].item(),
                        losses[1].item()))
                    self.writer.add_image('input', make_grid(img.detach().cpu(), nrow=8, normalize=True))
                    self.writer.add_image('generated', make_grid(fake_images.detach().cpu(), nrow=8, normalize=True))
                
            fake_images = self.models[0](self.fixed_noise).detach().cpu()
            self.writer.add_image('fixed_noise', make_grid(fake_images, padding=2, normalize=True))
            
        logs = [self.valid_metrics[idx].result() for idx in range(len(self.valid_metrics))] 
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
        
class WGAN_Trainer(Trainer):
    def __init__(self, models, criterion, metric_ftns, optimizers, config, device, device_ids,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(models, criterion, metric_ftns, optimizers, config, device, 
                         device_ids, data_loader, valid_data_loader, lr_scheduler, len_epoch)
        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        
        latent_dim = self.config['generator']['args']['latent_dim']
        
        for idx in range(len(self.models)):
            self.models[idx].train()
        
        for idx in range(len(self.train_metrics)):
            self.train_metrics[idx].reset()
            
            
        for batch_idx, (img, _, _) in enumerate(self.data_loader):
            losses = [None, None]
            batch_size = img.size(0)
            
            # real images
            real_imgs = img.to(self.device) 
            
            # one-sided label smoothing
            real_labels = torch.ones(batch_size, device=self.device)
            
            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=self.device)  # format NCHW
            fake_images = self.models[0](noise)
            
            fake_labels = -real_labels # fake label = -1   
            # flipped_fake_labels = real_labels # here, fake label = 1
            
            
            # train discriminator
            self.optimizers[1].zero_grad()

            real_preds = self.models[1](real_imgs).view(-1)
            real_loss = self.criterion(real_preds, real_labels)
            
            fake_preds = self.models[1](fake_images.detach()).view(-1)
            fake_loss = self.criterion(fake_preds, fake_labels)
            
            losses[1] = 0.5 * (real_loss + fake_loss)
            losses[1].backward()
            self.optimizers[1].step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            gradsum = 0.0
            for p in self.models[1].parameters():
                if p.grad is not None:
                    gradsum += torch.sum(p.grad.detach().cpu() ** 2)
            if gradsum > 1e-5:
                self.train_metrics[6].update('D_grad', gradsum.item())

            
            # clamp discriminator weights
            for p in self.models[1].parameters():
                p.data.clamp_(-0.01, 0.01)
            
            # train generator
            self.optimizers[0].zero_grad()
            
            fake_preds = self.models[1](fake_images).view(-1)
            losses[0] = self.criterion(fake_preds, real_labels)
            losses[0].backward()
            self.optimizers[0].step()

            gradsum = 0.0
            for p in self.models[0].parameters():
                if p.grad is not None:
                    gradsum += torch.sum(p.grad.detach().cpu() ** 2)
            if gradsum > 1e-5:
                self.train_metrics[7].update('G_grad', gradsum.item())


            self.train_metrics[0].update('G_loss', losses[0].item())
            self.train_metrics[1].update('D_loss', losses[1].item())
            self.train_metrics[2].update('D_real', real_loss.item())
            self.train_metrics[3].update('D_fake', fake_loss.item())
            
            pred_labels_real = torch.where(real_preds.detach() > 0., 1., -1.)
            pred_labels_fake = torch.where(fake_preds.detach() > 0., 1., -1.)
            
            self.train_metrics[4].update('real_acc', (pred_labels_real == real_labels).float().mean() * 100.)
            self.train_metrics[5].update('fake_acc', (pred_labels_fake == fake_labels).float().mean() * 100.)
            
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
        
        logs = [self.train_metrics[idx].result() for idx in range(len(self.train_metrics))] 
        
        ## validation
        """
        for idx in range(len(self.models)):
            self.models[idx].eval()
            
        with torch.no_grad():
            fake_images = self.models[0](self.fixed_noise).detach().cpu()
            self.writer.add_image('fixed_noise', make_grid(fake_images, padding=2, normalize=True))
        """
        if self.do_validation:
            val_logs = self._valid_epoch(epoch)
            
            for log_t, log_v in zip(logs, val_logs):
                log_t.update(**{'val_'+k : v for k, v in log_v.items()})
        
        """
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        """
        return logs
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        for idx in range(len(self.models)):
            self.models[idx].eval()
            
        for idx in range(len(self.valid_metrics)):
            self.valid_metrics[idx].reset()
        
        latent_dim = self.config['generator']['args']['latent_dim']
        
        with torch.no_grad():
            for batch_idx, (img, _, _) in enumerate(self.valid_data_loader):
                
                losses = [None, None]
                batch_size = img.size(0)
                
                # real images
                real_imgs = img.to(self.device) 
                real_labels = torch.ones(batch_size, device=self.device)
                
                # generated (fake) images
                noise = torch.randn(batch_size, latent_dim, 1, 1, device=self.device)  # format NCHW
                fake_images = self.models[0](noise)
                fake_labels = -real_labels
                
                real_preds = self.models[1](real_imgs).view(-1)
                real_loss = self.criterion(real_preds, real_labels)
                
                fake_preds = self.models[1](fake_images.detach()).view(-1)
                fake_loss = self.criterion(fake_preds, fake_labels)
                
                losses[1] = 0.5 * (real_loss + fake_loss)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                gradsum = 0.0
                for p in self.models[1].parameters():
                    if p.grad is not None:
                        gradsum += torch.sum(p.grad.detach().cpu() ** 2)
                if gradsum > 1e-5:
                    self.valid_metrics[6].update('D_grad', gradsum.item())

                # clamp discriminator weights
                for p in self.models[1].module.net.parameters():
                    p.data.clamp_(-0.01, 0.01)
                    
                # validate generator
                fake_preds = self.models[1](fake_images).view(-1)
                losses[0] = self.criterion(fake_preds, fake_labels)

                gradsum = 0.0
                for p in self.models[0].parameters():
                    if p.grad is not None:
                        gradsum += torch.sum(p.grad.detach().cpu() ** 2)
                if gradsum > 1e-5:
                    self.valid_metrics[7].update('G_grad', gradsum.item())

                self.valid_metrics[0].update('G_loss', losses[0].item())
                self.valid_metrics[1].update('D_loss', losses[1].item())
                self.valid_metrics[2].update('D_real', real_loss.item())
                self.valid_metrics[3].update('D_fake', fake_loss.item())
                
                pred_labels_real = torch.where(real_preds.detach() > 0., 1., -1.)
                pred_labels_fake = torch.where(fake_preds.detach() > 0., 1., -1.)
                self.valid_metrics[4].update('real_acc', (pred_labels_real == real_labels).float().mean() * 100.)
                self.valid_metrics[5].update('fake_acc', (pred_labels_fake == fake_labels).float().mean() * 100.)
                
                if batch_idx % self.log_step == 0:
                    self.logger.debug('Val Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx, mode='valid'),
                        losses[0].item(),
                        losses[1].item()))
                    self.writer.add_image('generated', make_grid(fake_images.detach().cpu(), nrow=8, normalize=True))
                
            fake_images = self.models[0](self.fixed_noise).detach().cpu()
            self.writer.add_image('fixed_noise', make_grid(fake_images, padding=2, normalize=True))
            
        logs = [self.valid_metrics[idx].result() for idx in range(len(self.valid_metrics))] 
            
            
        """
        
        # add histogram of model parameters to the tensorboard
        for model in self.models:
            for name, p in model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        """
        return logs
    
    
class WGAN_GP_Trainer(Trainer):
    def __init__(self, models, criterion, metric_ftns, optimizers, config, device, device_ids,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(models, criterion, metric_ftns, optimizers, config, device, 
                         device_ids, data_loader, valid_data_loader, lr_scheduler, len_epoch)

        self.train_metrics.append(MetricTracker('GP_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer))
        self.valid_metrics.append(MetricTracker('GP_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer))
        
        self.discr_iter_per_generator_iter = self.config['trainer']['discr_iter_per_generator_iter']
        self.epsilon = self.config['trainer']['epsilon']
        self.gradient_penalty_weight = self.config['trainer']['gradient_penalty_weight']
        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        
        latent_dim = self.config['generator']['args']['latent_dim']
        
        for idx in range(len(self.models)):
            self.models[idx].train()
        
        for idx in range(len(self.train_metrics)):
            self.train_metrics[idx].reset()
            
        skip_generator = 1
        for batch_idx, (img, _, _) in enumerate(self.data_loader):
            losses = [None, None]
            batch_size = img.size(0)
            
            # real images
            real_images = img.to(self.device) 
            
            # one-sided label smoothing
            real_labels = torch.ones(batch_size, device=self.device)
            
            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=self.device)  # format NCHW
            fake_images = self.models[0](noise)
            
            fake_labels = -real_labels # fake label = -1   
            # flipped_fake_labels = real_labels # here, fake label = 1
            
            
            # train discriminator
            self.optimizers[1].zero_grad()

            real_preds = self.models[1](real_images).view(-1)
            real_loss = self.criterion(real_preds, real_labels)
            
            fake_preds = self.models[1](fake_images.detach()).view(-1)
            fake_loss = self.criterion(fake_preds, fake_labels)
            
            losses[1] = 0.5 * (real_loss + fake_loss)
            
            # gradient clipping
            alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
            interpolated = alpha * real_images + (1 - alpha) * fake_images.detach()
            interpolated.requires_grad = True


            outputs = self.models[1](interpolated)
            grad_values = torch.ones(outputs.size()).to(self.device)
            
            gradients = torch.autograd.grad(
                    outputs=outputs,
                    inputs=interpolated,
                    grad_outputs=grad_values,
                    create_graph=True,
                    retain_graph=True)[0]
            
            gradients = gradients.view(batch_size, -1)
            
            # calc. norm of gradients, adding epsilon to prevent 0 values
            epsilon = 1e-13
            gradients_norm = torch.sqrt(
                torch.sum(gradients ** 2, dim=1) + self.epsilon)

            gp_penalty_term = ((gradients_norm - 1) ** 2).mean() * self.gradient_penalty_weight
            losses[1] += gp_penalty_term
            self.train_metrics[6].update('GP_loss', gp_penalty_term.item())
            
            losses[1].backward()
            self.optimizers[1].step()
            
            if skip_generator <= self.discr_iter_per_generator_iter:
                
                # train generator
                self.optimizers[0].zero_grad()
                
                fake_preds = self.models[1](fake_images).view(-1)
                losses[0] = self.criterion(fake_preds, real_labels)
                losses[0].backward()
                self.optimizers[0].step()
                
                skip_generator += 1
                
            else:
                skip_generator = 1
                losses[0] = torch.tensor(0.)
        

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics[0].update('G_loss', losses[0].item())
            self.train_metrics[1].update('D_loss', losses[1].item())
            self.train_metrics[2].update('D_real', real_loss.item())
            self.train_metrics[3].update('D_fake', fake_loss.item())
            
            pred_labels_real = torch.where(real_preds.detach() > 0., 1., 0.)
            pred_labels_fake = torch.where(fake_preds.detach() > 0., 1., 0.)
            
            self.train_metrics[4].update('real_acc', (pred_labels_real == (real_labels + 0.1)).float().mean() * 100.)
            self.train_metrics[5].update('fake_acc', (pred_labels_fake == fake_labels).float().mean() * 100.)
            
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
        
        logs = [self.train_metrics[idx].result() for idx in range(len(self.train_metrics))] 
        
        ## validation
        """
        for idx in range(len(self.models)):
            self.models[idx].eval()
            
        with torch.no_grad():
            fake_images = self.models[0](self.fixed_noise).detach().cpu()
            self.writer.add_image('fixed_noise', make_grid(fake_images, padding=2, normalize=True))
        """
        if self.do_validation:
            val_logs = self._valid_epoch(epoch)
            
            for log_t, log_v in zip(logs, val_logs):
                log_t.update(**{'val_'+k : v for k, v in log_v.items()})
        
        """
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        """
        return logs
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        for idx in range(len(self.models)):
            self.models[idx].eval()
            
        for idx in range(len(self.valid_metrics)):
            self.valid_metrics[idx].reset()
        
        latent_dim = self.config['generator']['args']['latent_dim']
        
        with torch.no_grad():
            for batch_idx, (img, _, _) in enumerate(self.valid_data_loader):
                
                losses = [None, None]
                batch_size = img.size(0)
                
                # real images
                real_images = img.to(self.device) 
                
                # one-sided label smoothing
                real_labels = torch.ones(batch_size, device=self.device)
                
                # generated (fake) images
                noise = torch.randn(batch_size, latent_dim, 1, 1, device=self.device)  # format NCHW
                fake_images = self.models[0](noise)
                
                fake_labels = -real_labels # fake label = -1   
                # flipped_fake_labels = real_labels # here, fake label = 1
                
                
                # train discriminator
                real_preds = self.models[1](real_images).view(-1)
                real_loss = self.criterion(real_preds, real_labels)
                
                fake_preds = self.models[1](fake_images.detach()).view(-1)
                fake_loss = self.criterion(fake_preds, fake_labels)
                
                losses[1] = 0.5 * (real_loss + fake_loss)
                
                # gradient clipping
                alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
                interpolated = alpha * real_images + (1 - alpha) * fake_images.detach()
                interpolated.requires_grad = True


                outputs = self.models[1](interpolated)
                grad_values = torch.ones(outputs.size()).to(self.device)
                
                gradients = torch.autograd.grad(
                        outputs=outputs,
                        inputs=interpolated,
                        grad_outputs=grad_values,
                        create_graph=True,
                        retain_graph=True)[0]
                
                gradients = gradients.view(batch_size, -1)
                
                # calc. norm of gradients, adding epsilon to prevent 0 values
                gradients_norm = torch.sqrt(
                    torch.sum(gradients ** 2, dim=1) + self.epsilon)

                gp_penalty_term = ((gradients_norm - 1) ** 2).mean() * self.gradient_penalty_weight
                losses[1] += gp_penalty_term
                self.valid_metrics[6].update('GP_loss', gp_penalty_term.item())
                
                if skip_generator <= self.discr_iter_per_generator_iter:
                    
                    fake_preds = self.models[1](fake_images).view(-1)
                    losses[0] = self.criterion(fake_preds, real_labels)
                    
                    skip_generator += 1
                    
                else:
                    skip_generator = 1
                    losses[0] = torch.tensor(0.)
            

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics[0].update('G_loss', losses[0].item())
                self.valid_metrics[1].update('D_loss', losses[1].item())
                self.valid_metrics[2].update('D_real', real_loss.item())
                self.valid_metrics[3].update('D_fake', fake_loss.item())
                
                pred_labels_real = torch.where(real_preds.detach() > 0., 1., 0.)
                pred_labels_fake = torch.where(fake_preds.detach() > 0., 1., 0.)
                self.valid_metrics[4].update('real_acc', (pred_labels_real == real_labels).float().mean() * 100.)
                self.valid_metrics[5].update('fake_acc', (pred_labels_fake == fake_labels).float().mean() * 100.)
                
                if batch_idx % self.log_step == 0:
                    self.logger.debug('Val Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx, mode='valid'),
                        losses[0].item(),
                        losses[1].item()))
                    self.writer.add_image('generated', make_grid(fake_images.detach().cpu(), nrow=8, normalize=True))
                
            fake_images = self.models[0](self.fixed_noise).detach().cpu()
            self.writer.add_image('fixed_noise', make_grid(fake_images, padding=2, normalize=True))
            
        logs = [self.valid_metrics[idx].result() for idx in range(len(self.valid_metrics))] 
            
            
        """
        
        # add histogram of model parameters to the tensorboard
        for model in self.models:
            for name, p in model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        """
        return logs
    

class AE_Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, models, criterion, metric_ftns, optimizers, config, device, device_ids,
                 data_loader, valid_data_loader=None, lr_schedulers=None, len_epoch=None):
        super().__init__(models, criterion, metric_ftns, optimizers, config)
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
        # self.do_validation = True
        self.lr_schedulers = lr_schedulers
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = [MetricTracker('encoder_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('decoder_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('l1_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)]

        self.valid_metrics = [MetricTracker('encoder_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('decoder_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('l1_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)]

        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        
        for idx in range(len(self.models)):
            self.models[idx].train()
        
        for idx in range(len(self.train_metrics)):
            self.train_metrics[idx].reset()
            
            
        for batch_idx, (img, _, _) in enumerate(self.data_loader):

            # real images
            real_imgs = img.to(self.device) 

            outputs = self.models[0](real_imgs)
            recons = self.models[1](outputs)

            
            self.optimizers[0].zero_grad()
            self.optimizers[1].zero_grad()
            
            loss = self.criterion(recons, real_imgs)
            loss.backward()
            
            self.optimizers[0].step()
            self.optimizers[1].step()
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            # print discriminator gradients
            gradsum = 0.0
            for p in self.models[0].parameters():
                if p.grad is not None:
                    gradsum += torch.sum(p.grad.detach().cpu() ** 2)
            if gradsum > 1e-5:
                self.train_metrics[0].update('encoder_grad', gradsum.item())
            
            # print generator gradients
            gradsum = 0.0
            for p in self.models[1].parameters():
                if p.grad is not None:
                    gradsum += torch.sum(p.grad.detach().cpu() ** 2)
            if gradsum > 1e-5:
                self.train_metrics[1].update('decoder_grad', gradsum.item())

            self.train_metrics[2].update('l1_loss', loss.item())
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} LR: {} L1_Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    self.lr_schedulers[0].get_last_lr()[0],
                    loss.item()))
                self.writer.add_image('input', make_grid(img.detach().cpu(), nrow=8, normalize=True))
                self.writer.add_image('recons', make_grid(recons.detach().cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        
        logs = [self.train_metrics[idx].result() for idx in range(len(self.train_metrics))] 
        
        ## validation
        if self.do_validation:
            val_logs = self._valid_epoch(epoch)
            
            for log_t, log_v in zip(logs, val_logs):
                log_t.update(**{'val_'+k : v for k, v in log_v.items()})
        
        if self.lr_schedulers is not None:
            self.lr_schedulers[0].step()
            self.lr_schedulers[1].step()

        return logs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        for idx in range(len(self.models)):
            self.models[idx].eval()
            
        for idx in range(len(self.valid_metrics)):
            self.valid_metrics[idx].reset()
        
        
        with torch.no_grad():
            for batch_idx, (img, _, _) in enumerate(self.valid_data_loader):
                
                # real images
                real_imgs = img.to(self.device) 
                outputs = self.models[0](real_imgs)
                recons = self.models[1](outputs)
                
                loss = self.criterion(recons, real_imgs)
                
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                gradsum = 0.0
                for p in self.models[0].parameters():
                    if p.grad is not None:
                        gradsum += torch.sum(p.grad.detach().cpu() ** 2)
                if gradsum > 1e-5:
                    self.valid_metrics[0].update('encoder_grad', gradsum.item())

                gradsum = 0.0
                for p in self.models[1].parameters():
                    if p.grad is not None:
                        gradsum += torch.sum(p.grad.detach().cpu() ** 2)
                if gradsum > 1e-5:
                    self.valid_metrics[1].update('decoder_grad', gradsum.item())
                self.valid_metrics[2].update('l1_loss', loss.item())
                
                if batch_idx % self.log_step == 0:
                    self.logger.debug('Val Epoch: {} {} L1_loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx, mode='valid'),
                        loss.item()))
                    self.writer.add_image('input', make_grid(real_imgs.detach().cpu(), nrow=8, normalize=True))
                    self.writer.add_image('recons', make_grid(recons.detach().cpu(), nrow=8, normalize=True))
                
            
        logs = [self.valid_metrics[idx].result() for idx in range(len(self.valid_metrics))] 
            
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
        
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        encoder_state = {
            'type': type(self.models[0].module).__name__ if type(self.models[0]).__name__ == 'DataParallel' else type(self.models[0]).__name__ , 
            'state_dict': self.models[0].state_dict(),
            'optimizer': self.optimizers[0].state_dict(),
            'monitor_best': self.mnt_best[0],
            'config': self.config
        }
        
        decoder_state = {
            'type': type(self.models[1].module).__name__ if type(self.models[1]).__name__ == 'DataParallel' else type(self.models[1]).__name__, 
            'state_dict': self.models[1].state_dict(),
            'optimizer': self.optimizers[1].state_dict(),
            'monitor_best': self.mnt_best[1],
            'config': self.config
        }
        
        state = {
            'epoch': epoch,
            'encoder': encoder_state,
            'decoder': decoder_state
        }
        
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = [checkpoint['encoder']['monitor_best'], checkpoint['decoder']['monitor_best']]

        # load architecture params from checkpoint.
        # load generator
        try:
            if checkpoint['encoder']['type'] != self.config['encoder']['type']:
                self.logger.warning("Warning: Architecture configuration (Encoder) given in config file is different from that of "
                                    "checkpoint. This may yield an exception while state_dict is being loaded.")
            self.models[0].load_state_dict(checkpoint['encoder']['state_dict'])
        except KeyError as e:
            self.logger.warning(e)
            self.logger.warning("Warning: Encoder state_dict not found in checkpoint. Encoder model not being resumed.")
        
        # load discriminator
        try:
            if checkpoint['decoder']['type'] != self.config['decoder']['type']:
                self.logger.warning("Warning: Architecture configuration (Decoder) given in config file is different from that of "
                                    "checkpoint. This may yield an exception while state_dict is being loaded.")
            self.models[1].load_state_dict(checkpoint['decoder']['state_dict'])
        except KeyError as e:
            self.logger.warning(e)
            self.logger.warning("Warning: Decoder state_dict not found in checkpoint. Decoder model not being resumed.")
         
        # load generator optimizer            
        try:
            if checkpoint['encoder']['config']['optimizer']['type'] != self.config['optimizer']['type']:
                self.logger.warning("Warning: Optimizer type (Encoder) given in config file is different from that of checkpoint. "
                                    "Optimizer parameters not being resumed.")
            self.optimizers[0].load_state_dict(checkpoint['encoder']['optimizer'])
        except KeyError as e:
            self.logger.warning(e)
            self.logger.warning("Warning: Encoder optimizer not found in checkpoint. Encoder optimizer not being resumed.")
        except ValueError as e:
            self.logger.warning(e)
            self.logger.warning("Warning: Encoder optimizer parameters not being resumed.")
        
        # load discriminator optimizer
        try:
            if checkpoint['decoder']['config']['optimizer']['type'] != self.config['optimizer']['type']:
                self.logger.warning("Warning: Optimizer type (Decoder) given in config file is different from that of checkpoint. "
                                    "Optimizer parameters not being resumed.")
            self.optimizers[1].load_state_dict(checkpoint['decoder']['optimizer'])
        except KeyError as e:
            self.logger.warning(e)
            self.logger.warning("Warning: Decoder optimizer not found in checkpoint. Decoder optimizer not being resumed.")
        except ValueError as e:
            self.logger.warning(e)
            self.logger.warning("Warning: Decoder optimizer parameters not being resumed.")
            
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def evaluate(self):
        print('Evaluating autoencoder...')
        xx = np.linspace(4240, 4340, 10000)
        
        with torch.no_grad():
            
            pbar = tqdm(total=len(self.data_loader) * self.data_loader.batch_size, desc="Evaluating nights...")
            for batch_idx, (img, _, _) in enumerate(self.data_loader):
                
                # create folder for batch
                batch_path = os.path.join(OUTPUT_FOLDER, f'batch_{batch_idx}')
                if not Path(batch_path).exists():
                    Path(batch_path).mkdir(parents=True, exist_ok=True)
                
                real_imgs = img.to(self.device)

                outputs = self.models[0](real_imgs)
                recons = self.models[1](outputs)
    
                # cut last 644 pixels
                recons  = torch.reshape(recons, (recons.shape[0], -1))
                recons = recons[:, :-644]
                recons = torch.reshape(recons, (recons.shape[0], 58, 10000))
                
                real_imgs = torch.reshape(real_imgs, (real_imgs.shape[0], -1))
                real_imgs = real_imgs[:, :-644]
                real_imgs = torch.reshape(real_imgs, (real_imgs.shape[0], 58, 10000))
                
                # save npz
                np.savez_compressed(os.path.join(batch_path, f'batch_{batch_idx}.npz'), 
                                    real_imgs=np.array(real_imgs.detach().cpu().numpy(), dtype=np.float32),
                                    recons=np.array(recons.detach().cpu().numpy(), dtype=np.float32),
                                    l1_loss=self.criterion(recons, real_imgs).item(),
                                    mse_loss=F.mse_loss(recons, real_imgs).item())
                
                for obs_idx, (real_spec, inpainted_spec) in enumerate(zip(real_imgs, recons)):
                    
                    # create folder for batch
                    obs_path = os.path.join(batch_path, f'obs_{obs_idx}')
                    if not Path(obs_path).exists():
                        Path(obs_path).mkdir(parents=True, exist_ok=True)

                    for idx, (real, inpainted) in enumerate(zip(real_spec, inpainted_spec)):
                        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                        ax.plot(xx, inpainted.detach().cpu().numpy(), color='black', linewidth=0.5, label='Generated')
                        ax.plot(xx, real.detach().cpu().numpy(), color='red', linewidth=0.5, alpha=0.6, label='Original')
                        ax.set_xlabel('Wavelength')
                        ax.set_ylabel('Normalized Flux between [-1, 1]')
                        # ax.set_title('Generated spectrum')
                        ax.grid()
                        ax.legend()
                        
                        out_fname = os.path.join(obs_path, f'generated_spectrum_{idx}.png')
                        fig.savefig(out_fname, dpi=300)
                        plt.close(fig)
                        
                    pbar.update(1)
                if batch_idx == 0:
                    break


class AAE_Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, models, criterion, metric_ftns, optimizers, config, device, device_ids,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(models, criterion, metric_ftns, optimizers, config)
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
        # self.do_validation = True
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = [MetricTracker('G_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_real', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_fake', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('G_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('real_acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('fake_acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer),]

        self.valid_metrics = [MetricTracker('G_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_real', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_fake', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('G_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('real_acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('fake_acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer),]

        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        
        latent_dim = self.config['generator']['args']['latent_dim']
        
        for idx in range(len(self.models)):
            self.models[idx].train()
        
        for idx in range(len(self.train_metrics)):
            self.train_metrics[idx].reset()
            
        for batch_idx, (img, _, _) in enumerate(self.data_loader):
            losses = [None, None]
            batch_size = img.size(0)

            valid = torch.ones(batch_size, 1, device=self.device)
            fake = torch.zeros(batch_size, 1, device=self.device)
            real_imgs = img.to(self.device) 

            # train generator
            self.optimizers[0].zero_grad()

            decoded_imgs, encoded_imgs = self.models[0](real_imgs)
            dgz = self.models[1](encoded_imgs)

            losses[0] = self.criterion[0](real_imgs, decoded_imgs, dgz, valid)
            losses[0].backward()
            self.optimizers[0].step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # print generator gradients
            gradsum = 0.0
            for p in self.models[0].parameters():
                if p.grad is not None:
                    gradsum += torch.sum(p.grad.detach().cpu() ** 2)
            if gradsum > 1e-5:
                self.train_metrics[4].update('G_grad', gradsum.item())

            # train discriminator
            self.optimizers[1].zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=self.device)  # format NCHW

            real_preds = self.models[1](noise)
            real_loss = self.criterion[1](real_preds, valid)

            fake_preds = self.models[1](encoded_imgs.detach())
            fake_loss = self.criterion[1](fake_preds, fake)
            
            losses[1] = 0.5 * (real_loss + fake_loss)
            losses[1].backward()
            self.optimizers[1].step()
            

            # print discriminator gradients
            gradsum = 0.0
            for p in self.models[1].parameters():
                if p.grad is not None:
                    gradsum += torch.sum(p.grad.detach().cpu() ** 2)
            if gradsum > 1e-5:
                self.train_metrics[5].update('D_grad', gradsum.item())
            

            self.train_metrics[0].update('G_loss', losses[0].item())
            self.train_metrics[1].update('D_loss', losses[1].item())
            self.train_metrics[2].update('D_real', real_loss.item())
            self.train_metrics[3].update('D_fake', fake_loss.item())
            
            pred_labels_real = torch.where(F.sigmoid(real_preds.detach()) > 0.5, 1., 0.)
            pred_labels_fake = torch.where(F.sigmoid(fake_preds.detach()) > 0.5, 1., 0.)
            
            self.train_metrics[6].update('real_acc', (pred_labels_real == valid).float().mean() * 100.)
            self.train_metrics[7].update('fake_acc', (pred_labels_fake == fake).float().mean() * 100.)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    losses[0].item(),
                    losses[1].item()))
                
            if batch_idx == self.len_epoch:
                break
        
        logs = [self.train_metrics[idx].result() for idx in range(len(self.train_metrics))] 
        
        if self.do_validation:
            val_logs = self._valid_epoch(epoch)
            
            for log_t, log_v in zip(logs, val_logs):
                log_t.update(**{'val_'+k : v for k, v in log_v.items()})
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return logs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        for idx in range(len(self.models)):
            self.models[idx].eval()
            
        for idx in range(len(self.valid_metrics)):
            self.valid_metrics[idx].reset()
        
        latent_dim = self.config['generator']['args']['latent_dim']
        
        with torch.no_grad():
            for batch_idx, (img, _, _) in enumerate(self.valid_data_loader):
                
                losses = [None, None]
                batch_size = img.size(0)

                valid = torch.ones(batch_size, 1, device=self.device)
                fake = torch.zeros(batch_size, 1, device=self.device)
                real_imgs = img.to(self.device) 

                decoded_imgs, encoded_imgs = self.models[0](real_imgs)
                dgz = self.models[1](encoded_imgs)
                losses[0] = self.criterion[0](real_imgs, decoded_imgs, dgz, valid)
                
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # print generator gradients
                gradsum = 0.0
                for p in self.models[0].parameters():
                    if p.grad is not None:
                        gradsum += torch.sum(p.grad.detach().cpu() ** 2)
                if gradsum > 1e-5:
                    self.valid_metrics[4].update('G_grad', gradsum.item())

                # train discriminator
                noise = torch.randn(batch_size, latent_dim, device=self.device)  # format NCHW
                real_preds = self.models[1](noise)
                real_loss = self.criterion[1](real_preds, valid)
                fake_preds = self.models[1](encoded_imgs.detach())
                fake_loss = self.criterion[1](fake_preds, fake)
                losses[1] = 0.5 * (real_loss + fake_loss)

                # print discriminator gradients
                gradsum = 0.0
                for p in self.models[1].parameters():
                    if p.grad is not None:
                        gradsum += torch.sum(p.grad.detach().cpu() ** 2)
                if gradsum > 1e-5:
                    self.valid_metrics[5].update('D_grad', gradsum.item())
                

                self.valid_metrics[0].update('G_loss', losses[0].item())
                self.valid_metrics[1].update('D_loss', losses[1].item())
                self.valid_metrics[2].update('D_real', real_loss.item())
                self.valid_metrics[3].update('D_fake', fake_loss.item())
                
                pred_labels_real = torch.where(F.sigmoid(real_preds.detach()) > 0.5, 1., 0.)
                pred_labels_fake = torch.where(F.sigmoid(fake_preds.detach()) > 0.5, 1., 0.)
            
                self.valid_metrics[6].update('real_acc', (pred_labels_real == valid).float().mean() * 100.)
                self.valid_metrics[7].update('fake_acc', (pred_labels_fake == fake).float().mean() * 100.)


                if batch_idx % self.log_step == 0:
                    self.logger.debug('Val Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx, mode='valid'),
                        losses[0].item(),
                        losses[1].item()))
                    self.writer.add_image('input', make_grid(img.detach().cpu(), nrow=8, normalize=True))
                    self.writer.add_image('decoded', make_grid(decoded_imgs.detach().cpu(), nrow=8, normalize=True))
                
            
        logs = [self.valid_metrics[idx].result() for idx in range(len(self.valid_metrics))] 
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
    
    def evaluate(self):
        print('Evaluating Adversarial Autoencoder...')
        xx = np.linspace(4240, 4340, 10000)

        for idx in range(len(self.models)):
            self.models[idx].eval()
        
        with torch.no_grad():
            
            pbar = tqdm(total=len(self.data_loader) * self.data_loader.batch_size, desc="Evaluating nights...")
            for batch_idx, (img, _, _) in enumerate(self.data_loader):
                
                # create folder for batch
                batch_path = os.path.join(OUTPUT_FOLDER, f'batch_{batch_idx}')
                if not Path(batch_path).exists():
                    Path(batch_path).mkdir(parents=True, exist_ok=True)
                
                real_imgs = img.to(self.device)

                # we only need the encoder and decoder (generator)
                batch_size = img.size(0)

                valid = torch.ones(batch_size, 1, device=self.device)
                fake = torch.zeros(batch_size, 1, device=self.device)
                decoded_imgs, encoded_imgs = self.models[0](real_imgs)
                dgz = self.models[1](encoded_imgs)
                G_loss = self.criterion[0](real_imgs, decoded_imgs, dgz, valid)
                fake_preds = self.models[1](encoded_imgs.detach())
                D_loss = self.criterion[1](fake_preds, fake)

                self.logger.debug('Evaluating: {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                    self._progress(batch_idx, mode='train'),
                    G_loss.item(),
                    D_loss.item()))
    
                # cut last 644 pixels
                decoded_imgs  = torch.reshape(decoded_imgs, (decoded_imgs.shape[0], -1))
                decoded_imgs = decoded_imgs[:, :-644]
                decoded_imgs = torch.reshape(decoded_imgs, (decoded_imgs.shape[0], 58, 10000))
                
                real_imgs = torch.reshape(real_imgs, (real_imgs.shape[0], -1))
                real_imgs = real_imgs[:, :-644]
                real_imgs = torch.reshape(real_imgs, (real_imgs.shape[0], 58, 10000))
                
                # save npz
                np.savez_compressed(os.path.join(batch_path, f'batch_{batch_idx}.npz'), 
                                    real_imgs=np.array(real_imgs.detach().cpu().numpy(), dtype=np.float32),
                                    decoded_imgs=np.array(decoded_imgs.detach().cpu().numpy(), dtype=np.float32),
                                    G_loss=G_loss.item(),
                                    D_loss=D_loss.item(),
                                    l1_loss=F.l1_loss(decoded_imgs, real_imgs).item(),
                                    mse_loss=F.mse_loss(decoded_imgs, real_imgs).item())
                
                for obs_idx, (real_spec, inpainted_spec) in enumerate(zip(real_imgs, decoded_imgs)):
                    
                    # create folder for batch
                    obs_path = os.path.join(batch_path, f'obs_{obs_idx}')
                    if not Path(obs_path).exists():
                        Path(obs_path).mkdir(parents=True, exist_ok=True)

                    for idx, (real, inpainted) in enumerate(zip(real_spec, inpainted_spec)):
                        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                        ax.plot(xx, inpainted.detach().cpu().numpy(), color='black', linewidth=0.5, label='Generated')
                        ax.plot(xx, real.detach().cpu().numpy(), color='red', linewidth=0.5, alpha=0.6, label='Original')
                        ax.set_xlabel('Wavelength')
                        ax.set_ylabel('Normalized Flux between [-1, 1]')
                        # ax.set_title('Generated spectrum')
                        ax.grid()
                        ax.legend()
                        
                        out_fname = os.path.join(obs_path, f'generated_spectrum_{idx}.png')
                        fig.savefig(out_fname, dpi=300)
                        plt.close(fig)
                        
                    pbar.update(1)
                if batch_idx == 0:
                    break


class CGAN_Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, models, criterion, metric_ftns, optimizers, config, device, device_ids,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(models, criterion, metric_ftns, optimizers, config)
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
        # self.do_validation = True
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = [MetricTracker('G_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_real', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_fake', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('real_acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('fake_acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('G_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer)]

        self.valid_metrics = [MetricTracker('G_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_real', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_fake', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('real_acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('fake_acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('D_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer),
                              MetricTracker('G_grad', *[m.__name__ for m in self.metric_ftns], writer=self.writer)]

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        
        latent_dim = self.config['generator']['args']['latent_dim']
        
        for idx in range(len(self.models)):
            self.models[idx].train()
        
        for idx in range(len(self.train_metrics)):
            self.train_metrics[idx].reset()
            
            
        for batch_idx, (img, params, _) in enumerate(self.data_loader):
            losses = [None, None]
            batch_size = img.size(0)
            
            # real images
            real_imgs = img.to(self.device) 
            params = params.to(self.device)

            # one-sided label smoothing
            real_labels = torch.ones(batch_size, device=self.device) * 0.9
            
            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=self.device)  # format NCHW
            fake_images = self.models[0](noise, params)
            
            fake_labels = torch.zeros(batch_size, device=self.device) # fake label = 0
            # flipped_fake_labels = real_labels # here, fake label = 1
            
            
            # train discriminator
            self.optimizers[1].zero_grad()

            real_preds = self.models[1](real_imgs, params).view(-1)
            real_loss = self.criterion(real_preds, real_labels)
            
            fake_preds = self.models[1](fake_images.detach(), params).view(-1)
            fake_loss = self.criterion(fake_preds, fake_labels)
            
            losses[1] = 0.5 * (real_loss + fake_loss)
            losses[1].backward()
            self.optimizers[1].step()
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            # print discriminator gradients
            gradsum = 0.0
            for p in self.models[1].parameters():
                if p.grad is not None:
                    gradsum += torch.sum(p.grad.detach().cpu() ** 2)
            if gradsum > 1e-5:
                self.train_metrics[6].update('D_grad', gradsum.item())
            
            # train generator
            self.optimizers[0].zero_grad()
            
            fake_preds = self.models[1](fake_images, params).view(-1)
            losses[0] = self.criterion(fake_preds, real_labels)
            losses[0].backward()
            self.optimizers[0].step()

            # print generator gradients
            gradsum = 0.0
            for p in self.models[0].parameters():
                if p.grad is not None:
                    gradsum += torch.sum(p.grad.detach().cpu() ** 2)
            if gradsum > 1e-5:
                self.train_metrics[7].update('G_grad', gradsum.item())

            self.train_metrics[0].update('G_loss', losses[0].item())
            self.train_metrics[1].update('D_loss', losses[1].item())
            self.train_metrics[2].update('D_real', real_loss.item())
            self.train_metrics[3].update('D_fake', fake_loss.item())
            
            pred_labels_real = torch.where(F.sigmoid(real_preds.detach()) > 0.5, 1., 0.)
            pred_labels_fake = torch.where(F.sigmoid(fake_preds.detach()) > 0.5, 1., 0.)
            
            self.train_metrics[4].update('real_acc', (pred_labels_real == (real_labels + 0.1)).float().mean() * 100.)
            self.train_metrics[5].update('fake_acc', (pred_labels_fake == fake_labels).float().mean() * 100.)
            
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

            if batch_idx == self.len_epoch:
                break
        
        logs = [self.train_metrics[idx].result() for idx in range(len(self.train_metrics))] 
        
        if self.do_validation:
            val_logs = self._valid_epoch(epoch)
            
            for log_t, log_v in zip(logs, val_logs):
                log_t.update(**{'val_'+k : v for k, v in log_v.items()})
        
        return logs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        for idx in range(len(self.models)):
            self.models[idx].eval()
            
        for idx in range(len(self.valid_metrics)):
            self.valid_metrics[idx].reset()
        
        latent_dim = self.config['generator']['args']['latent_dim']
        
        with torch.no_grad():
            for batch_idx, (img, params, _) in enumerate(self.valid_data_loader):
                
                losses = [None, None]
                batch_size = img.size(0)
                
                # real images
                real_imgs = img.to(self.device) 
                params = params.to(self.device)
                real_labels = torch.ones(batch_size, device=self.device)
                
                # generated (fake) images
                noise = torch.randn(batch_size, latent_dim, 1, 1, device=self.device)  # format NCHW
                fake_images = self.models[0](noise, params)
                fake_labels = torch.zeros(batch_size, device=self.device) # fake label = 0
                
                real_preds = self.models[1](real_imgs, params).view(-1)
                real_loss = self.criterion(real_preds, real_labels)
                
                fake_preds = self.models[1](fake_images.detach(), params).view(-1)
                fake_loss = self.criterion(fake_preds, fake_labels)
                
                losses[1] = 0.5 * (real_loss + fake_loss)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                gradsum = 0.0
                for p in self.models[1].parameters():
                    if p.grad is not None:
                        gradsum += torch.sum(p.grad.detach().cpu() ** 2)
                if gradsum > 1e-5:
                    self.valid_metrics[6].update('D_grad', gradsum.item())

                # validate generator
                fake_preds = self.models[1](fake_images, params).view(-1)
                losses[0] = self.criterion(fake_preds, fake_labels)

                gradsum = 0.0
                for p in self.models[0].parameters():
                    if p.grad is not None:
                        gradsum += torch.sum(p.grad.detach().cpu() ** 2)
                if gradsum > 1e-5:
                    self.valid_metrics[7].update('G_grad', gradsum.item())


                self.valid_metrics[0].update('G_loss', losses[0].item())
                self.valid_metrics[1].update('D_loss', losses[1].item())
                self.valid_metrics[2].update('D_real', real_loss.item())
                self.valid_metrics[3].update('D_fake', fake_loss.item())
                
                pred_labels_real = torch.where(F.sigmoid(real_preds.detach()) > 0.5, 1., 0.)
                pred_labels_fake = torch.where(F.sigmoid(fake_preds.detach()) > 0.5, 1., 0.)
                self.valid_metrics[4].update('real_acc', (pred_labels_real == real_labels).float().mean() * 100.)
                self.valid_metrics[5].update('fake_acc', (pred_labels_fake == fake_labels).float().mean() * 100.)
                
                if batch_idx % self.log_step == 0:
                    self.logger.debug('Val Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx, mode='valid'),
                        losses[0].item(),
                        losses[1].item()))
                    self.writer.add_image('input', make_grid(img.detach().cpu(), nrow=8, normalize=True))
                    self.writer.add_image('generated', make_grid(fake_images.detach().cpu(), nrow=8, normalize=True))
                
        logs = [self.valid_metrics[idx].result() for idx in range(len(self.valid_metrics))] 
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
        