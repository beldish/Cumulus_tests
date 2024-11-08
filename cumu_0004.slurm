#!/bin/bash
#SBATCH --job-name=test_04
#SBATCH --partition=gpu                    # Specify the GPU partition
#SBATCH --gres=gpu:1                       # Request 1 GPUs (generic resource)
#SBATCH --ntasks=1                         # Number of tasks (usually 1 for CUDA)
#SBATCH --cpus-per-task=4                  # CPU cores per task (for CUDA/GPU jobs)
#SBATCH --mem=16G                          # Memory per node
#SBATCH --time=02:00:00                    # Time limit for the job (hh:mm:ss)
#SBATCH --output=cuda_job_%j.out           # Output file (with job ID)

# Load necessary modules (CUDA, compiler, etc.)
module load CUDA/11.7.0                    # Load the CUDA toolkit (adjust version)


# Run the CUDA application
srun ./bin/test_0004                           # Repace with your actual CUDA program


