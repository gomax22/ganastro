# Adapted from https://github.com/victoresque/pytorch-template/blob/master/train.py

import argparse
import collections
import torch
import numpy as np
import data.loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import WGAN_Trainer
from utils.util import prepare_device


def main(config):
    logger = config.get_logger('train')
    print("Logger is set up.")

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    print("Data loader is set up.")

    # build model architecture, then print to console
    generator = config.init_obj('generator', module_arch)
    print("Generator is set up.")
    discriminator = config.init_obj('discriminator', module_arch)
    print("Discriminator is set up.")
    
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
    
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    generator_trainable_params = filter(lambda p: p.requires_grad, generator.parameters())
    discriminator_trainable_params = filter(lambda p: p.requires_grad, discriminator.parameters())
    generator_optimizer = config.init_obj('optimizer', torch.optim, generator_trainable_params)
    discriminator_optimizer = config.init_obj('optimizer', torch.optim, discriminator_trainable_params)
    # generator_optimizer = torch.optim.Adam(generator_trainable_params, lr=0.0003, betas=(0.5, 0.999))
    # discriminator_optimizer = torch.optim.Adam(discriminator_trainable_params, lr=0.0001, betas=(0.5, 0.999))
    
    optimizers = [generator_optimizer, discriminator_optimizer]

    trainer = WGAN_Trainer(models, criterion, metrics, optimizers,
                      config=config,
                      device=device,
                      device_ids=device_ids,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader
    )
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Training GANASTRO')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    """
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    """
    config = ConfigParser.from_args(args)
    main(config)