#!/bin/bash -l

# Job general details
#SBATCH --job-name captain3plot-dev
#SBATCH --account <YOUR_ACCOUNT>
#SBATCH --mail-user <YOUR_EMAIL>
#SBATCH --mail-type NONE
#SBATCH --time 00:29:59

# Paths and output
#SBATCH --output logs/captain3plot-dev-%j.out

# Ressources
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 1

# Environment
#SBATCH --export NONE

export SLURM_EXPORT_ENV=ALL

VENV_PATH=~/captain3-alps/.venv
RUN_SCRIPT=~/captain3preview/examples/plot_input_data.py

EDF_FILE=./ngc-pytorch-24.06.toml

srun --environment ${EDF_FILE} /bin/bash -c '\
    '"${VENV_PATH}"'/bin/python '"${RUN_SCRIPT}"''

echo THE END
