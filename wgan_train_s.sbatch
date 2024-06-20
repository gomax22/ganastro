#!/bin/bash
#SBATCH --job-name=wganastro
#SBATCH --output=scripts/wganastro_l.out
#SBATCH --error=scripts/wganastro_l.err
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xgpu
#SBATCH --mem=88000

python wgan_train.py --config wgan_l.json