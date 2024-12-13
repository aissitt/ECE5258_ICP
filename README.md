# ECE5258_ICP

This repository contains code for training and evaluating deep learning models for hemodynamic simulations, specifically focusing on Left Ventricular Assist Device (LVAD) and aneurysm data. The repository supports models with both data-driven and physics-informed loss functions using a 3D U-Net architecture.

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aissitt/ECE5258_ICP.git
   ```

2. **Set Output Directory and Configure Environment Variables** (if using `use_env_vars` in `config.json`):
   Do this in the `train_multigpu.sh` and `eval_multigpu.sh` files:
   ```bash
   OUTPUT_BASE_DIR=/path/to/your/ECE5258_ICP/training_outputs
   export INPUT_DATA_PATH=/path/to/input/data
   export OUTPUT_DATA_PATH=/path/to/output/data
   ```

## Training and Evaluation

### Training

To train a model, use the `train_multigpu.sh` script with SLURM:

```bash
sbatch train_multigpu.sh data  # For data-driven model
sbatch train_multigpu.sh physics      # For physics-informed model
```

Training parameters like epochs, batch size, learning rate, and data paths are configured in the `config.json` file.

### Evaluation

To evaluate a model, use the `eval_multigpu.sh` script with SLURM:

```bash
sbatch eval_multigpu.sh data  # For data-driven model
sbatch eval_multigpu.sh physics      # For physics-informed model
```
