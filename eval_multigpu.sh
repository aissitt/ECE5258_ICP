#!/bin/bash

# Manpage: https://slurm.schedmd.com/sbatch.html

##################################
######### Configuration ##########
##################################

##################################
####### Resources Request ########
##################################

# Use GPU partition (gpu1 and gpu2) or other partition (e.g.: short)
#SBATCH --partition=gpu1

# Configure the number of nodes (in partition above)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

# Configure the number of GPUs to use all available GPUs on the node
#SBATCH --gpus-per-node=4
# Set time limit 3 days
#SBATCH --time=3-00:00:00
#SBATCH --job-name=LVAD_Eval_MultiGPU

# Prepare a timestamp for directory naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse the evaluation type argument (data or physics)
EVAL_TYPE=$1

# Validate EVAL_TYPE
if [[ "$EVAL_TYPE" != "data" && "$EVAL_TYPE" != "physics" ]]; then
    echo "Invalid evaluation type specified. Use 'data' or 'physics'."
    exit 1
fi

# Define the base directory for the training runs
OUTPUT_BASE_DIR="/home1/aissitt2019/Hemodynamics/LVAD/training_outputs"

# Find the most recent training run directory for the specified mode
TRAIN_RUN_DIR=$(ls -dt ${OUTPUT_BASE_DIR}/${EVAL_TYPE}/train_run_* | head -n 1)

# Create directories for this evaluation run under the corresponding training run folder
RUN_DIR=${TRAIN_RUN_DIR}/eval_run_${TIMESTAMP}
mkdir -p "${RUN_DIR}/images"
mkdir -p "${RUN_DIR}/logs"

# Direct output and error files to the run-specific directory
#SBATCH --output=${RUN_DIR}/LVAD_Eval_MultiGPU_%j.out
#SBATCH --error=${RUN_DIR}/LVAD_Eval_MultiGPU_%j.err

echo "Starting evaluation job on $(hostname) at $(date)"

# Load conda environment named 'hemodynamics'
. ./env.sh hemodynamics2

# Move to the LVAD directory containing the scripts
cd /home1/aissitt2019/Hemodynamics/LVAD

# Set environment variables for data paths
export INPUT_DATA_PATH="/home1/aissitt2019/LVAD/LVAD_data/lvad_rdfs_inlets.npy"     # Expects shape (x, 128, 128, 128, 2)
export OUTPUT_DATA_PATH="/home1/aissitt2019/LVAD/LVAD_data/lvad_vels.npy"           # Expects shape (x, 128, 128, 128, 3)

# Optional: Specify a model path argument
MODEL_PATH_ARG=$2

# Check if a specific model path is provided; if not, default to the most recent model
if [[ -z "$MODEL_PATH_ARG" ]]; then
    MODEL_PATH="${TRAIN_RUN_DIR}/lvad_model_*.h5"
    
    # Find the most recent model file for evaluation
    LATEST_MODEL=$(ls -t "$MODEL_PATH" 2>/dev/null | head -n 1)
    if [[ -z "$LATEST_MODEL" ]]; then
        echo "No model file found matching pattern $MODEL_PATH"
        exit 1
    fi
else
    # Use the provided model path argument
    LATEST_MODEL=$MODEL_PATH_ARG
fi

# Optional: Specify a config file argument
CONFIG_PATH_ARG=$3

# Track total runtime
start=$(date +%s)

# Build the python command with or without config based on the argument
if [[ -z "$CONFIG_PATH_ARG" ]]; then
    # No config file specified, run eval without it
    python eval.py --model-path "$LATEST_MODEL" --mode "$EVAL_TYPE" --output-dir "$RUN_DIR"
else
    # Config file specified, include it in the eval command
    python eval.py --model-path "$LATEST_MODEL" --mode "$EVAL_TYPE" --output-dir "$RUN_DIR" --config "$CONFIG_PATH_ARG"
fi

end=$(date +%s)

runtime=$((end-start))
echo "Runtime: $runtime seconds"
echo "Evaluation job completed at $(date)"
