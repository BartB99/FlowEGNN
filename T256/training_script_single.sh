#!/bin/bash
#SBATCH -J T256-OVF
#SBATCH -t 12:00:00
#SBATCH -p gpu_a100
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1   
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bart.bronsgeest@surf.nl

set -euxo pipefail

export MPLCONFIGDIR=$TMPDIR/matplotlib
export XDG_CACHE_HOME=$TMPDIR/.cache
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export WANDB_HTTP_TIMEOUT=60

module load 2023
# module load Python/3.11.5
source /gpfs/home4/bartb/venvs/boids/bin/activate

torchrun --standalone --nproc_per_node=1 src/train_ddp.py "$@"