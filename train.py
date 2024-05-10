from trainers.dcgan_trainer import DCGAN_Trainer
import argparse
import json
import os
import torch

def check_args(args):
    if not os.path.exists(args['data_root']):
        raise Exception(f"Data root directory {args['data_root']} does not exist")
    if not os.path.exists(args['config']):
        raise Exception(f"Config file {args['config']} does not exist")

    return args

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train GANASTRO model')
    ap.add_argument('--data-root', required=True, type=str, help='Path to the data root directory')
    ap.add_argument('--config', required=True, type=str, help='Path to the configuration file')
    ap.add_argument('--use-cuda', required=False, type=bool, default=True, help='cuda-enabled training')
   
    args = vars(ap.parse_args())
    args = check_args(args)
    
    data_root = args['data_root']
    config = args['config']
    use_cuda = args['use_cuda']
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    
    
    with open(config, 'r') as f:
        cfg = json.load(f)
    
    trainer = DCGAN_Trainer(data_root=data_root, config=cfg, device=device)
    trainer.train() 