#!/bin/bash

#SBATCH -o gpuexp-4-2.log-%j
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Initialize the module command first source
source /etc/profile

# Load cuda and Python Module
module load anaconda/2022b            # Load Anaconda module or specific Python module
module load cuda/11.4                   # Load CUDA module

# Call your script as you would from the command line
python gpuexp-4-2.py