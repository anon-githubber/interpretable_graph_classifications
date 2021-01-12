#!/bin/bash

#SBATCH --partition=SCSEGPU_UG 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M 
#SBATCH --job-name=job 

module load anaconda
source /home/FYP/heyu0012/.bashrc
conda activate GCNN_GAP_graphgen
conda env list

# MODEL=GCN
MODEL=DFScodeRNN_cls

# DATA=MUTAG
# DATA=NCI-H23
DATA=TOX21_AR
# DATA=PTC_FR


ROOT=/home/FYP/heyu0012/projects/interpretable_graph_classifications

BASE_PATH=${ROOT}/data/${DATA}/

export MODEL=${MODEL}
export DATA=${DATA}
export BASE_PATH=$BASE_PATH

# add graphgen bin for generating dfs code
export PATH=${ROOT}/models/graphgen/bin:/$PATH

cd $ROOT

CUDA_VISIBLE_DEVICES=0 python main.py cuda=1 -gm=$MODEL -data=$DATA -retrain=1
