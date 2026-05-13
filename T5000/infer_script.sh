#!/bin/bash
#SBATCH -J infer
#SBATCH -t 4:00:00
#SBATCH -p gpu_h100
#SBATCH --nodes 1
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bart.bronsgeest@surf.nl

#Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

#Activate environment
source /gpfs/home4/bartb/venvs/boids/bin/activate

python /gpfs/home4/bartb/T256/T256-SUBBOX/src/infer.py "$@"
