#!/bin/bash

job_directory=$PWD/.job
mkdir -p $job_directory
mkdir -p /scratch/jain.sar/runs
job_file="${job_directory}/$1.job"

echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40Gb
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/jain.sar/runs/$1.%j.out
#SBATCH --error=/scratch/jain.sar/runs/$1.%j.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1
export CUDA_DEVICE=0
export NER_INF_BASEDIR=/scratch/jain.sar/ner_influence
python ${@:2}" > $job_file

sbatch $job_file

echo ${@:2}

rm $job_file
