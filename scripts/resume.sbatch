#!/bin/bash
#SBATCH --job-name=ganastro
#SBATCH --output=resume.out
#SBATCH --error=resume.err
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xgpu
#SBATCH --mem=88000

python train.py --resume <path-to-checkpoint> --config <path-to-config>