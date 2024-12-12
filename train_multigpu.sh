#!/bin/bash

# Manpage: https://slurm.schedmd.com/sbatch.html

##################################
######### Configuration ##########
##################################

##################################
####### Resources Request ########
##################################

#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=4
#SBATCH --time=3-00:00:00
#SBATCH --job-name=LVAD_Train_MultiGPU

# Prepare a timestamp for directory naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Mode-specific subfolder (data or physics)
TRAIN_TYPE=$1

# Shift to handle additional arguments like --use-tuning or --seed
shift

# Validate TRAIN_TYPE
if [[ "$TRAIN_TYPE" != "data" && "$TRAIN_TYPE" != "physics" ]]; then
    echo "Invalid training type specified. Use 'data' or 'physics'."
    exit 1
fi

# Define the base directory for training runs
OUTPUT_BASE_DIR=/home1/aissitt2019/Hemodynamics/LVAD/training_outputs

# Define the run directory at the top level, including the timestamp
RUN_DIR=${OUTPUT_BASE_DIR}/${TRAIN_TYPE}/train_run_${TIMESTAMP}

# Create directories for this training run
mkdir -p ${RUN_DIR}/logs
mkdir -p ${RUN_DIR}/images

# SLURM output and error files (log paths)
#SBATCH --output=${RUN_DIR}/LVAD_Train_MultiGPU_%j.out
#SBATCH --error=${RUN_DIR}/LVAD_Train_MultiGPU_%j.err

echo "Starting training job on $(hostname) at $(date)"

# Load the conda environment
. ./env.sh hemodynamics2

# Move to the LVAD directory containing the scripts
cd /home1/aissitt2019/Hemodynamics/LVAD

# Set environment variables for data paths
export INPUT_DATA_PATH="/home1/aissitt2019/LVAD/LVAD_data/lvad_rdfs_inlets.npy"
export OUTPUT_DATA_PATH="/home1/aissitt2019/LVAD/LVAD_data/lvad_vels.npy"

# export NCCL_DEBUG=INFO

start=$(date +%s)

# Run the training script with additional arguments passed through "$@"
python train.py --output-dir ${RUN_DIR} --mode ${TRAIN_TYPE} "$@"

end=$(date +%s)

runtime=$((end-start))
echo "Runtime: $runtime seconds"
echo "Training job completed at $(date)"
