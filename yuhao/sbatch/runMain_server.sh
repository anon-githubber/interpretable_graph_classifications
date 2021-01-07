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

DATA=MUTAG

ROOT=/home/FYP/heyu0012/projects/interpretable_graph_classifications
GRAPHGEN_RESULTS_PATH=/home/FYP/heyu0012/results/interpretable_graph_classifications/graphgen/DfsRNN_${DATA}_classification/
mkdir -p $GRAPHGEN_RESULTS_PATH
export GRAPHGEN_RESULTS_PATH=$GRAPHGEN_RESULTS_PATH

MUTAG_DATASET_PATH=/home/FYP/heyu0012/results/graphgen/DfsRNN_MUTAG_classification/datasets/MUTAG
MUTAG_LABEL_PATH=$MUTAG_DATASET_PATH
MUTAG_DFSTENSOR_PATH=${MUTAG_DATASET_PATH}/min_dfscode_tensors
export MUTAG_LABEL_PATH=$MUTAG_LABEL_PATH
export MUTAG_DFSTENSOR_PATH=$MUTAG_DFSTENSOR_PATH

cd $ROOT

CUDA_VISIBLE_DEVICES=0 python main.py cuda=1 -gm=$MODEL -data=$DATA -retrain=1
