#!/bin/bash
#SBATCH --job-name=BMIaligner
#SBATCH --output=/n/data1/hms/neurobio/sabatini/gyu/github_clone/Mothership_Zeta/slurm_out/slurm-%A_%a.out
#SBATCH -p short
#SBATCH  -t 5:00:00
#SBATCH --mem=200G
#SBATCH -c 1

module load matlab/2021b
matlab -nodesktop -batch "addpath(\"/n/data1/hms/neurobio/sabatini/gyu/github_clone/BWAIN/notebooks\"); addpath(\"/n/data1/hms/neurobio/sabatini/gyu/github_clone/BWAIN/helper_functions\"); Align_logger_to_movie_slurm(\""$@"\")"

