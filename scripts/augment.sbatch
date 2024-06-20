#!/bin/bash
#SBATCH --job-name=augment
#SBATCH --output=augment.out
#SBATCH --error=augment.err
#SBATCH --ntasks=1                                                              
#SBATCH --nodes=1                                                               
#SBATCH --partition=xhicpu
#SBATCH --cpus-per-task=32                                              
#SBATCH --mem=88000  

# python ../augment.py --dataset /projects/data/HARPN/K9/20180708 --output /projects/data/HARPN/K9_augmented2 --samples-per-night 58 --window-size 5 --num-filters 10000 --cutoff-begin 4240.0 --cutoff-end 4340.0 --concurrency True
# python ../augment.py --dataset /projects/data/HARPN/K9/20180722 --output /projects/data/HARPN/K9_augmented2 --samples-per-night 58 --window-size 5 --num-filters 10000 --cutoff-begin 4240.0 --cutoff-end 4340.0 --concurrency True
# python ../augment.py --dataset /projects/data/HARPN/K9/20180723 --output /projects/data/HARPN/K9_augmented2 --samples-per-night 58 --window-size 5 --num-filters 10000 --cutoff-begin 4240.0 --cutoff-end 4340.0 --concurrency True
# python ../augment.py --dataset /projects/data/HARPN/K9/20180901 --output /projects/data/HARPN/K9_test --samples-per-night 58 --window-size 5 --num-filters 10000 --cutoff-begin 4240.0 --cutoff-end 4340.0 --concurrency True
# python ../augment.py --dataset /projects/data/HARPN/K9/20180904 --output /projects/data/HARPN/K9_augmented2 --samples-per-night 58 --window-size 5 --num-filters 10000 --cutoff-begin 4240.0 --cutoff-end 4340.0 --concurrency True
# python ../augment.py --dataset /projects/data/HARPN/K9/20210905 --output /projects/data/HARPN/K9_augmented2 --samples-per-night 58 --window-size 5 --num-filters 10000 --cutoff-begin 4240.0 --cutoff-end 4340.0 --concurrency True

# python ../augment.py --dataset /projects/data/HARPN/K9/20180708 --output /projects/data/HARPN/K9_augmented_reduced --samples-per-night 58 --window-size 5 --num-filters 5000 --cutoff-begin 4240.0 --cutoff-end 4340.0 --concurrency True
# python ../augment.py --dataset /projects/data/HARPN/K9/20180722 --output /projects/data/HARPN/K9_augmented_reduced --samples-per-night 58 --window-size 5 --num-filters 5000 --cutoff-begin 4240.0 --cutoff-end 4340.0 --concurrency True
# python ../augment.py --dataset /projects/data/HARPN/K9/20180723 --output /projects/data/HARPN/K9_augmented_reduced --samples-per-night 58 --window-size 5 --num-filters 5000 --cutoff-begin 4240.0 --cutoff-end 4340.0 --concurrency True
# python ../augment.py --dataset /projects/data/HARPN/K9/20180901 --output /projects/data/HARPN/K9_test_reduced --samples-per-night 58 --window-size 5 --num-filters 5000 --cutoff-begin 4240.0 --cutoff-end 4340.0 --concurrency True
# python ../augment.py --dataset /projects/data/HARPN/K9/20180904 --output /projects/data/HARPN/K9_augmented_reduced --samples-per-night 58 --window-size 5 --num-filters 5000 --cutoff-begin 4240.0 --cutoff-end 4340.0 --concurrency True
python ../augment.py --dataset /projects/data/HARPN/K9/20210905 --output /projects/data/HARPN/K9_augmented_reduced --samples-per-night 58 --window-size 5 --num-filters 5000 --cutoff-begin 4240.0 --cutoff-end 4340.0 --concurrency True
