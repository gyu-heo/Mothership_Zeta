#!/bin/bash
#SBATCH --job-name=S_seqNMF
#SBATCH --output=/n/data1/hms/neurobio/sabatini/gyu/github_clone/Mothership_Zeta/slurm_out/seqNMF/slurm-%A_%a.out
#SBATCH -t 0-05:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH -c 2
set -e

module load gcc/6.2.0
module load conda2/4.2.13
module load cuda/11.2

source activate /home/gyh930/.conda/envs/ROICaT

cd /n/data1/hms/neurobio/sabatini/gyu/github_clone
echo "$@"
python3 -c "import Mothership_Zeta as MZ; MZ.BMI_IDAP_process.seqNMF_single_process()" "$@"