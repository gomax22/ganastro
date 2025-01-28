#!/bin/bash
#SBATCH --job-name=ganpainting
#SBATCH --output=inpaint.out
#SBATCH --error=inpaint.err
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xgpu
#SBATCH --mem=88000

python ../inpaint.py --resume ../pretrained/0920_170346/checkpoint-epoch250.pth --config ../inpainting.json