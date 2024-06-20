import torch
import torch.nn as nn
import torch.optim as optim
from torch.functional import F
import numpy as np
import argparse
import collections
import torch
import numpy as np
import data.loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Inpainter
from utils.util import prepare_device

"""
class EarlyStopping:
    def __init__(self, convergence_threshold=1e-5, patience=10):
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.best_loss = float('inf')
        self.no_improvement_steps = 0

    def step(self, loss):
        if abs(self.best_loss - loss) < self.convergence_threshold:
            self.no_improvement_steps += 1
        else:
            self.no_improvement_steps = 0

        self.best_loss = loss

        if self.no_improvement_steps >= self.patience:
            return True
        return False
"""

def main(args):
    logger = config.get_logger('inpaint')
    print("Logger is set up.")

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    print("Data loader is set up.")

    # build model architecture, then print to console
    generator = config.init_obj('generator', module_arch)
    print("Generator is set up.")
    discriminator = config.init_obj('discriminator', module_arch)
    
    logger.info(generator)
    logger.info(discriminator)
    print("Model architecture is set up.")

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    print("Device is set up.")
    generator, discriminator = generator.to(device), discriminator.to(device)
    
    if len(device_ids) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=device_ids)
        discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)

    models = [generator, discriminator]
    
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # get zhat and its optimizer
    parameters = nn.Parameter(torch.FloatTensor(data_loader.batch_size, generator.module.latent_dim, 1, 1).uniform_(-1, 1).to(device))
    logger.info(f"Parameters shape: {parameters.shape}")
    optimizer = config.init_obj('optimizer', torch.optim, [parameters])
    
    inpainter = Inpainter(models, parameters, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      device_ids=device_ids,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader
    )
    inpainter.train()
    
    
"""
    for i, (corrupted_images, original_images, masks, weighted_masks) in enumerate(dataloader):
        corrupted_images, masks, weighted_masks = corrupted_images.cuda(), masks.cuda(), weighted_masks.cuda()
        z_optimum = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1, (corrupted_images.shape[0],args.latent_dim,))).cuda())
        optimizer_inpaint = optim.Adam([z_optimum])

        print("Starting backprop to input ...")
        for epoch in range(args.optim_steps):
            optimizer_inpaint.zero_grad()
            generated_images = generator(z_optimum)
            discriminator_opinion = discriminator(generated_images)
            c_loss = context_loss(corrupted_images, generated_images, weighted_masks)
            prior_loss = torch.sum(-torch.log(discriminator_opinion))
            inpaint_loss = c_loss + args.prior_weight*prior_loss
            inpaint_loss.backward()
            optimizer_inpaint.step()
            print("[Epoch: {}/{}] \t[Loss: \t[Context: {:.3f}] \t[Prior: {:.3f}] \t[Inpaint: {:.3f}]]  \r".format(1+epoch, args.optim_steps, c_loss, 
                                                                               prior_loss, inpaint_loss),end="")
            
        print("")
"""

if __name__ == '__main__':
    
    args = argparse.ArgumentParser(description='Inpainting with GANASTRO')
    args.add_argument('-c', '--config', default=None, type=str, required=True,
                      help='config file path (default: None)')
    args.add_argument('--checkpoint', default=None, required=False,
                      help='path to checkpoint file for loading parameters')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)
    main(config)
    
    