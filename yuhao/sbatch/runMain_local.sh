# module load anaconda
# source /home/FYP/heyu0012/.bashrc
# conda activate GCNN_GAP_graphgen
# conda env list

# MODEL=GCN
MODEL=DFScodeRNN_cls

DATA=NCI-H23

ROOT=/home/FYP/heyu0012/projects/interpretable_graph_classifications

BASE_PATH=${ROOT}/data/${DATA}/
# GRAPHGEN_RESULTS_PATH=/home/FYP/heyu0012/results/interpretable_graph_classifications/graphgen/DfsRNN_${DATA}_classification/
# rm -rf $BASE_PATH/graphgen

export MODEL=${MODEL}
export DATA=${DATA}
export BASE_PATH=$BASE_PATH

# add graphgen bin for generating dfs code
export PATH=${ROOT}/models/graphgen/bin:/$PATH

# MUTAG_DATASET_PATH=/home/FYP/heyu0012/results/graphgen/DfsRNN_MUTAG_classification/datasets/MUTAG
# MUTAG_LABEL_PATH=$MUTAG_DATASET_PATH
# MUTAG_DFSTENSOR_PATH=${MUTAG_DATASET_PATH}/min_dfscode_tensors
# export MUTAG_LABEL_PATH=$MUTAG_LABEL_PATH
# export MUTAG_DFSTENSOR_PATH=$MUTAG_DFSTENSOR_PATH

cd $ROOT

python main.py cuda=1 -gm=$MODEL -data=$DATA -retrain=1
