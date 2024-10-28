<S-Insert>#!/bin/bash

#Provide your prefered job name you can identify on the cluster queue
#SBATCH --job-name=example

# Define number of nodes needed, in out example its 1
# Each node has 32 CPU cores.
#SBATCH --nodes=1

#Define output files and error files
#SBATCH --output=%j_output.txt #%j  represents the jobid 
#SBATCH --error=%j_error.txt   #%j  represents the jobid


# For advanced tuning you can define number of tasks with --ntasks-per-* check "man sbatch" for details.
#SBATCH --ntasks-per-node=32

# Define estimated amount of time that you think your job will run.please bear in mind
# that if the time is shorter than the actual time the job runs the scheduler will terminate
# the job before it completes. If you put too much time it might take a bit longer for your job 
# run depending on resource availability.
#              d-hh:mm:ss
#SBATCH --time=0-00:05:00

# Define partition, this is optional
#SBATCH --partition cpu

# How much memory you need - this is optional as well 
# --mem =  memory per node 
# --mem-per-cpu = memory per CPU/core  - choose one
#SBATCH --mem-per-cpu=2000MB
##SBATCH --mem=100GB    # no effect beacuse of double hash

# Activate email notifications for more information
# For more values, check "man sbatch"
### SBATCH --mail-type=START,END,FAIL #This is not functional yet fo cumulus

# Define and/or create scratch/working directory 
SCRATCH_DIRECTORY=/lustre/home/your_username/
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# Copy input files to scratch/working directory
# ${SLURM_SUBMIT_DIR} points to the path where this script was submitted from
cp ${SLURM_SUBMIT_DIR}/myfiles*.txt ${SCRATCH_DIRECTORY}

# This is where you work is done, you can load modules and run command
time sleep 10

# for example
# module load openmpi-4


# After the job is done we copy our output back to $SLURM_SUBMIT_DIR
cp ${SCRATCH_DIRECTORY}/my_output ${SLURM_SUBMIT_DIR}


# Save everything to home directory and delete the work directory
cd ${SLURM_SUBMIT_DIR}
rm -rf ${SCRATCH_DIRECTORY}

# Finish the script
exit 0


