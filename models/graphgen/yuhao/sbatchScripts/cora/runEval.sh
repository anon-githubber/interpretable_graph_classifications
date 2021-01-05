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

GraphgenResultsPath='/home/FYP/heyu0012/results/graphgen/DfsRNN_cora/'
GraphgenModelPath=${GraphgenResultsPath}model_save/DFScodeRNN_cora_2020-12-13-09-38-59/DFScodeRNN_cora_400.dat

export GraphgenResultsPath=$GraphgenResultsPath
export GraphgenModelPath=$GraphgenModelPath
export PATH=/home/FYP/heyu0012/projects/graphgen/bin:$PATH

ROOT=/home/FYP/heyu0012/projects/graphgen
cd $ROOT
python evaluate.py
