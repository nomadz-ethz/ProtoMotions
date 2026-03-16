#!/bin/bash
#SBATCH --job-name=k1_mimic_kick
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=100G

# Load modules
module load eth_proxy

# Job info
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Extract container to local scratch (CRITICAL for performance)
echo "Extracting container..."
time tar -xzf /cluster/work/rsl/$USER/containers/protomotions-k1-app.tar.gz -C $TMPDIR

# Setup directories
RESULTS_DIR="/cluster/project/rsl/$USER/results/$SLURM_JOB_ID"
mkdir -p $RESULTS_DIR

# Run container
echo "Running application..."
time singularity exec \
    --nv \
    --bind $RESULTS_DIR:/output \
    --bind /cluster/scratch/$USER:/data:ro \
    $TMPDIR/protomotions-k1.sif \
    /isaac-sim/python.sh /workspace/protomotions/k1_kick/k1_mimic_kick.py \
        --simulator isaaclab \
        --num-envs 32 \
        --epochs 30 \
        --headless

echo "Job completed at $(date)"
