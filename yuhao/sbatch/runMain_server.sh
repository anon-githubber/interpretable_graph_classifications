#!/bin/bash

#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=q_ug8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M
#SBATCH --job-name=job

module load anaconda
source /home/FYP/heyu0012/.bashrc
conda activate GCNN_GAP_graphgen
# conda env list

# MODEL=GCN
MODEL=DFScodeRNN_cls
RNN_TYPE=GRU
DATA=MUTAG
# DATA=NCI-H23
# DATA=TOX21_AR
# DATA=PTC_FR

epochs=(50)
number_of_rnn_layers=(1 2)
embedding_sizes=(4 8 16)
hidden_sizes=(4 8)
number_of_mlp_layers=(1)
learning_rates=(0.0003 0.001 0.003 0.01)

ROOT=/home/FYP/heyu0012/projects/interpretable_graph_classifications

BASE_PATH=${ROOT}/data/${DATA}/

export MODEL=${MODEL}
export DATA=${DATA}
export BASE_PATH=$BASE_PATH

# add graphgen bin for generating dfs code
export PATH=${ROOT}/models/graphgen/bin:/$PATH

cd $ROOT

for epoch in "${epochs[@]}"
do
    for number_of_rnn_layer in "${number_of_rnn_layers[@]}"
    do
        for embedding_size in "${embedding_sizes[@]}"
        do
            for hidden_size in "${hidden_sizes[@]}"
            do 
                for number_of_mlp_layer in "${number_of_mlp_layers[@]}"
                do
                    for learning_rate in "${learning_rates[@]}"
                    do
                        CUDA_VISIBLE_DEVICES=0 python main.py cuda=1 -gm=$MODEL -data=$DATA -retrain=1 -params_turning=1 -rnn_type=$RNN_TYPE\
                            -epoch=$epoch\
                            -number_of_rnn_layer=$number_of_rnn_layer\
                            -embedding_size=$embedding_size\
                            -hidden_size=$hidden_size\
                            -number_of_mlp_layer=$number_of_mlp_layer\
                            -learning_rate=$learning_rate
                    done
                done
            done
        done
    done
done                   