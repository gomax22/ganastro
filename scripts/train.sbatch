#!/bin/bash
#SBATCH --job-name=ganastro
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xgpu
#SBATCH --mem=150000

python train.py --config dcgan.json
# python train.py --resume experiments/training/<path-to-checkpoint> --config experiments/training/<path-to-config>