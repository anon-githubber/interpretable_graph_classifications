#!/bin/bash
# cd /mnt/c/users/user/desktop/gitrepos/interpretable_graph_classification
# source /mnt/c/users/user/desktop/gitrepos/interpretable_graph_classification/venv/bin/activate
# ./run_model.sh
# input arguments
CUDA_VISIBLE_DEVICES=${GPU} python3 main.py cuda=1 -gm=DGCNN -data=MUTAG -retrain=1
