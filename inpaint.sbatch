#!/bin/bash
#SBATCH --job-name=ganpainting
#SBATCH --output=sscripts/inpaint_aug_real_8x8.out
#SBATCH --error=sscripts/inpaint_aug_fake_8x8.err
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xgpu
#SBATCH --mem=88000

python inpaint.py --config inpainting_real.json --resume inpaint/models/GANASTRO/0610_124318/checkpoint-epoch103.pth