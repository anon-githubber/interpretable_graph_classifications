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

GRAPHGEN_RESULTS_PATH='/home/FYP/heyu0012/results/graphgen/temp/'
mkdir -p $GRAPHGEN_RESULTS_PATH
export GRAPHGEN_RESULTS_PATH=$GRAPHGEN_RESULTS_PATH

ROOT=/home/FYP/heyu0012/projects/interpretable_graph_classifications/models/graphgen
export PATH=${ROOT}/bin:$PATH

cd $ROOT
cp -r ${ROOT}/datasets $GRAPHGEN_RESULTS_PATH
# ls $GraphgenResultsPath/datasets
python main.py
