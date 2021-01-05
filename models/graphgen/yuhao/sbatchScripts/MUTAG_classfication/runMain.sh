#!/bin/bash

#SBATCH --partition=SCSEGPU_UG 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M 
#SBATCH --job-name=job 

module load anaconda
source /home/FYP/heyu0012/.bashrc
conda activate graphgen
conda env list

GraphgenResultsPath='/home/FYP/heyu0012/results/graphgen/DfsRNN_MUTAG_classification/'
mkdir -p $GraphgenResultsPath
export GraphgenResultsPath=$GraphgenResultsPath
export PATH=/home/FYP/heyu0012/projects/graphgen/bin:$PATH

ROOT=/home/FYP/heyu0012/projects/graphgen
cd $ROOT
cp -r ${ROOT}/datasets $GraphgenResultsPath
# ls $GraphgenResultsPath/datasets
python main.py
