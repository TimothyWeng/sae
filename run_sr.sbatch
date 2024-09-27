#!/bin/bash
#SBATCH --job-name=symbolic_regression
#SBATCH --output=outputs/sr_output_%A_%a.out  # Output file for each job
#SBATCH --error=errors/sr_error_%A_%a.err    # Error file for each job
#SBATCH --time=6:00:00                # Adjust based on the expected runtime
#SBATCH --partition=all               # Use the correct partition
#SBATCH --gres=gpu:1                  # Request one GPU per job
#SBATCH --mem=32G                     # Adjust memory requirement
#SBATCH --cpus-per-task=32             # Adjust CPU count
#SBATCH --array=0-32                  # Job array for 33 jobs

# Compute the range for neurons in each job
NEURON_START=$(( SLURM_ARRAY_TASK_ID * 1000 ))
NEURON_END=$(( NEURON_START + 1000 ))

# Adjust for the last job (32,000 - 32,768 neurons)
if [ $SLURM_ARRAY_TASK_ID -eq 32 ]; then
    NEURON_END=32768
fi

# Run the Python script with the computed neuron range
source /grogu/user/mprabhud/miniconda3/etc/profile.d/conda.sh
conda activate pysr
python3 SR.py --neuron_start $NEURON_START --neuron_end $NEURON_END
