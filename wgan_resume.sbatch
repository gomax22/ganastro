#!/bin/bash
#SBATCH --job-name=wganastro
#SBATCH --output=scripts/wganastro_s.out
#SBATCH --error=scripts/wganastro_s.err
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xgpu
#SBATCH --mem=88000

python wgan_train.py --resume saved/models/GANASTRO/0527_163348/checkpoint-epoch16.pth --config saved/models/GANASTRO/0527_163348/config.json