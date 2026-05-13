#!/bin/bash
#SBATCH -J T256-SUBBOX
#SBATCH -t 50:00:00
#SBATCH -p gpu_h100
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bart.bronsgeest@surf.nl

set -euxo pipefail

export MPLCONFIGDIR=$TMPDIR/matplotlib
export XDG_CACHE_HOME=$TMPDIR/.cache
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

export TORCH_DISTRIBUTED_DEBUG=DETAIL

module load 2023
# module load Python/3.11.5
source /gpfs/home4/bartb/venvs/boids/bin/activate

torchrun --standalone --nproc_per_node=4 src/train_ddp.py "$@"