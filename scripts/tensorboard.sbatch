#!/bin/bash
#SBATCH --job-name="tensorboard"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

tensorboard --logdir=<log_dir> --bind_all --port 9100
