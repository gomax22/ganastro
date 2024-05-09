#!/bin/bash
#SBATCH --job-name=scramble
#SBATCH --output=scramble.out
#SBATCH --error=scramble.err
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xhicpu
#SBATCH --cpus-per-task=32                                              
#SBATCH --mem=88000  

python ../scramble.py --dataset /projects/data/HARPN/K9/20180722 --output /projects/data/HARPN/K9_synthetic --samples-per-night 58 --k 5 --max-nights 9750