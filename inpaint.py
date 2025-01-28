import torch
import torch.nn as nn
import argparse
import data.loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Inpainter
from utils.util import prepare_device


def main(config):
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
    inpainter.evaluate()


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
    args.add_argument('--output_dir', default=None, type=str, required=False,
                      help='directory to save outputs')
    config = ConfigParser.from_args(args)
    main(config)
