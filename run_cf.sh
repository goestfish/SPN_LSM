#!/bin/bash

#SBATCH --nodes=1               # node count

#SBATCH -p gpu --gres=gpu:1     # number of gpus per node

#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes

#SBATCH --cpus-per-task=4       # cpu-cores per task

#SBATCH -t 04:00:00             # total run time limit (HH:MM:SS)

#SBATCH --mem=32000MB           # memory

#SBATCH --job-name='SPN_LSM_CFs'

#SBATCH --output=slurm_logs/R-%x.%j.out

#SBATCH --error=slurm_logs/R-%x.%j.err



# Unbuffered python output

export PYTHONUNBUFFERED=1

export PYTHONIOENCODING=utf-8



module purge

unset LD_LIBRARY_PATH



# Bind your home/scratch/data into the container

export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"



# Same container you used before

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

echo "Running run_counterfactuals.py at $(date)"

echo "=========================================="

echo ""



# Go to your project directory

cd "${SLURM_SUBMIT_DIR}" || exit 1

echo "Working directory: $(pwd)"

echo ""



# If you ever need to (re)install deps inside the container, uncomment:

# $EXEC_PATH $CONTAINER_PATH pip install --user --no-cache-dir -r requirements.txt



# ---- ACTUAL COMMAND ----

$EXEC_PATH $CONTAINER_PATH python -u run_counterfactuals.py



EXIT_CODE=$?



echo ""

echo "=========================================="

echo "run_counterfactuals.py finished at $(date)"

echo "Exit code: $EXIT_CODE"

echo "=========================================="



exit $EXIT_CODE

