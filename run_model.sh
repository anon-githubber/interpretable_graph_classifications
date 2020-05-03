#!/bin/bash
# cd /mnt/c/users/sogge/desktop/gitrepos/explainable_graph_classification
# source /mnt/c/users/sogge/desktop/gitrepos/explainable_graph_classification/venv/bin/activate
# ./run_model.sh
# input arguments
CUDA_VISIBLE_DEVICES=${GPU} python3 main.py cuda=0 -gm=DGCNN -data=MUTAG
