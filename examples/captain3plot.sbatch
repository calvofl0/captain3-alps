#!/bin/bash -l

# Job general details
#SBATCH --job-name captain3plot
#SBATCH --account <YOUR_ACCOUNT>
#SBATCH --mail-user <YOUR_EMAIL>
#SBATCH --mail-type NONE
#SBATCH --time 00:29:59

# Paths and output
#SBATCH --output logs/captain3plot-%j.out

# Ressources
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 1

# Environment
#SBATCH --export NONE

export SLURM_EXPORT_ENV=ALL

CAPTAIN3_PATH=/usr/share/captain3
VENV_PATH=${CAPTAIN3_PATH}/venv
RUN_SCRIPT=${CAPTAIN3_PATH}/examples/plot_input_data.py

EDF_FILE=./ngc-pytorch-captain3preview-24.06.toml

srun --environment ${EDF_FILE} /bin/bash -c '\
    '"${VENV_PATH}"'/bin/python '"${RUN_SCRIPT}"''

echo THE END
