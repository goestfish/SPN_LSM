#!/bin/bash

#SBATCH --nodes=1               # node count

#SBATCH -p gpu --gres=gpu:1     # number of gpus per node

#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes

#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)

#SBATCH -t 08:00:00             # total run time limit (HH:MM:SS)

#SBATCH --mem=32000MB           # memory

#SBATCH --job-name='SPN_LSM_Analysis'

#SBATCH --output=slurm_logs/R-%x.%j.out

#SBATCH --error=slurm_logs/R-%x.%j.err



# Force unbuffered output

export PYTHONUNBUFFERED=1

export PYTHONIOENCODING=utf-8

module purge

unset LD_LIBRARY_PATH



export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"



# Same container as before

CONTAINER_PATH="/oscar/runtime/software/external/ngc-containers/tensorflow.d/x86_64.d/tensorflow-24.03-tf2-py3.simg"

EXEC_PATH="srun apptainer exec --nv"



echo ""

echo "=========================================="

echo "Job started at: $(date)"

echo "Job ID: $SLURM_JOB_ID"

echo "Node: $SLURM_NODELIST"

echo "=========================================="

echo ""



echo "GPU Information (from host):"

nvidia-smi || echo "nvidia-smi not available on host"

echo ""



echo "GPU Information (inside container):"

$EXEC_PATH $CONTAINER_PATH nvidia-smi || echo "nvidia-smi not available in container"

echo ""



echo "PyTorch GPU Detection (inside container):"

$EXEC_PATH $CONTAINER_PATH python - << 'EOF'

import torch

print("Torch version:", torch.__version__)

print("CUDA available:", torch.cuda.is_available())

print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():

    print("Current device:", torch.cuda.current_device())

    print("Device name:", torch.cuda.get_device_name(0))

EOF



echo ""

echo "=========================================="

echo "Installing Python dependencies (user-local)"

echo "=========================================="

echo ""



cd "${SLURM_SUBMIT_DIR}" || exit 1

echo "Working directory: $(pwd)"

echo ""



# Install from your repo's requirements file (recommended)

# Comment this out after the first run if install time is annoying.

# $EXEC_PATH $CONTAINER_PATH pip install --user --no-cache-dir -r requirements.txt



echo ""

echo "=========================================="

echo "Starting main_analysis.py at $(date)"

echo "=========================================="

echo ""



# ---- ACTUAL ANALYSIS COMMAND ----

# This model name must match the folder under cnn_spn_models/

MODEL_NAME="full_run_100e_ft25"



$EXEC_PATH $CONTAINER_PATH python -u main_analysis.py --filepath "$MODEL_NAME"



EXIT_CODE=$?



echo ""

echo "=========================================="

echo "Python script finished at $(date)"

echo "Exit code: $EXIT_CODE"

echo "=========================================="

exit $EXIT_CODE

