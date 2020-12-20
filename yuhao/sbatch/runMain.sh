#!/bin/bash

#SBATCH --partition=SCSEGPU_UG 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M 
#SBATCH --job-name=job 

module load anaconda
source /home/FYP/heyu0012/.bashrc
conda activate GCNN_GAP 
conda env list

ROOT=/home/FYP/heyu0012/projects/interpretable_graph_classifications
cd $ROOT

echo "PATH: " $PATH

CUDA_VISIBLE_DEVICES=0 python main.py cuda=1 -gm=GCN -data=MUTAG -retrain=1
