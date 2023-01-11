#!/bin/bash
#SBATCH --job-name=M_BMI_IDAP
#SBATCH --output=/n/data1/hms/neurobio/sabatini/gyu/github_clone/Mothership_Zeta/slurm_out/slurm-%A_%a.out
#SBATCH -t 0-01:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH -c 1
set -e

module load gcc/6.2.0
module load conda2/4.2.13
module load cuda/11.2

source activate /home/gyh930/.conda/envs/ROICaT

cd /n/data1/hms/neurobio/sabatini/gyu/github_clone
echo "$@"
python3 -c "import Mothership_Zeta as MZ; MZ.BMI_IDAP_process.BMI_IDAP_multi_process()" "$@"