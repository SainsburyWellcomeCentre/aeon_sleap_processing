#!/bin/bash

#SBATCH --job-name=sleap_train          # job name
#SBATCH --partition=gpu                 # partition (queue) # gpu # a100
#SBATCH --gres=gpu:1                    # number of gpus per node # a4500 # a100_2g.10gb #a100
#SBATCH --nodes=1                       # node count
#SBATCH --exclude=gpu-sr670-20          # DNN lib missing 
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --mem=256G                      # total memory per node 
#SBATCH --time=1-00:00:00               # total run time limit (DD-HH:MM:SS)
#SBATCH --output=slurm_output/%N_%j.out # output file path

mkdir -p slurm_output

python sleap_train.py -f aeon3_social02_ceph.slp --type centroid --use_split