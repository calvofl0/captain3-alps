#!/bin/bash -l

# Job general details
#SBATCH --job-name captain3inference_after_training
#SBATCH --account <YOUR_ACCOUNT>
#SBATCH --mail-user <YOUR_EMAIL>
#SBATCH --mail-type NONE
#SBATCH --time 00:29:59

# Paths and output
#SBATCH --output logs/captain3inference_after_training-%j.out

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
RUN_SCRIPT=${CAPTAIN3_PATH}/examples/run_inference.py

EXAMPLES_PATH=${CAPTAIN3_PATH}/examples
TRAINING_RESULTS_PATH=./output/training_results
INFERENCE_RESULTS_PATH=./output/inference_results
INFERENCE_RESULTS_AFTER_TRAINING_PATH=./output/inference_results_after_training

EDF_FILE=./ngc-pytorch-captain3preview-24.06.toml

sed_cmd=$(cat <<EOF
sed -i -e 's|\([[:blank:]]*TRAINED_MODEL[[:blank:]]*=[[:blank:]]*Path("\).*\(")\)|\1'"${TRAINING_RESULTS_PATH}/trained_weights.npy"'\2|' \
       -e 's|Path("'"${INFERENCE_RESULTS_PATH}"'")|Path("'"${INFERENCE_RESULTS_AFTER_TRAINING_PATH}"'")|' "${EXAMPLES_PATH}/run_inference.py"
EOF
)

srun --environment ${EDF_FILE} /bin/bash -c "
    $sed_cmd
    ${VENV_PATH}/bin/python ${RUN_SCRIPT}
"

echo THE END
