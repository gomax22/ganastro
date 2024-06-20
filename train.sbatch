#!/bin/bash
#SBATCH --job-name=ganastro
#SBATCH --output=sscripts/ganastro_aug_8x4.out
#SBATCH --error=sscripts/ganastro_aug_8x4.err
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xgpu
#SBATCH --mem=88000

python train.py --config dcgan.json