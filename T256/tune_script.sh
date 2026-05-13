#!/bin/bash
#SBATCH -J raytune_fm
#SBATCH -t 20:00:00
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bart.bronsgeest@surf.nl
#SBATCH --output=raytune-%j.out
#SBATCH --error=raytune-%j.err

set -euxo pipefail

export MPLCONFIGDIR=$TMPDIR/matplotlib
export XDG_CACHE_HOME=$TMPDIR/.cache
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

export RAY_TMPDIR="$TMPDIR/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

module load 2023
source /gpfs/home4/bartb/venvs/boids/bin/activate

cd /gpfs/home4/bartb/T256/T256-SUBBOX

python -c "
import torch, ray
print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, {torch.cuda.device_count()} GPUs')
print(f'Ray {ray.__version__}')
"

python src/train_tune.py \
    --config Configs/overfit_configs.yaml \
    --num_samples 20 \
    --tune_epochs 300 \
    --grace_period 50 \
    --gpus_per_trial 1.0

rm -rf "$RAY_TMPDIR"