#!/bin/bash -l
#
# Multithreading example job script for MPCDF Cobra.
# In addition to the Python example shown here, the script
# is valid for any multi-threaded program, including
# plain OpenMP, parallel Matlab, Julia, and similar cases.
#
#SBATCH -o /u/kroma/m4d/logs/multiverse_py_%j.out
#SBATCH -e /u/kroma/m4d/logs/multiverse_py_%j.err
#SBATCH -D ./
#SBATCH -J multiverse
#SBATCH --nodes=1             # request a full node
#SBATCH --ntasks-per-node=1   # only start 1 task via srun because Python multiprocessing starts more tasks internally
#SBATCH --cpus-per-task=72    # assign all the cores to that first task to make room for multithreading
#SBATCH --time=12:00:00
#SBATCH --mail-user=kessler@cbs.mpg.de
#SBATCH --mail-type=ALL

# # set number of OMP threads *per process*
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Check if the correct number of arguments is provided (e.g., 2 arguments)
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 experiment subject"
    exit 1
fi

# Access the arguments and set variables
experiment="$1"
subject="$2"

echo "Experiment: $experiment, Subject: $subject"

conda activate m4d
conda info --envs
cd /u/kroma/m4d
srun python3 /u/kroma/m4d/src/multiverse.py $experiment $subject