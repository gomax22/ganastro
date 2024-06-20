#!/bin/bash
#SBATCH --job-name=ganastro
#SBATCH --output=sscripts/ganastro_aug.out
#SBATCH --error=sscripts/ganastro_aug.err
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xgpu
#SBATCH --mem=88000

python train.py --resume ssaved/models/GANASTRO/0610_002517/checkpoint-epoch90.pth --config ssaved/models/GANASTRO/0610_002517/config.json