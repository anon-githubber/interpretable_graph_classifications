import random
import time
import pickle
from torch.utils.data import DataLoader

from args import Args
from utils import create_dirs
from datasets.process_dataset import create_graphs
from datasets.preprocess import calc_max_prev_node, dfscodes_weights
from baselines.dgmg.data import DGMG_Dataset_from_file
from baselines.graph_rnn.data import Graph_Adj_Matrix_from_file
# from graphgen.data import Graph_DFS_code_from_file
from graphgen_cls.data import Graph_DFS_code_from_file
from model import create_model
from train import train
from sklearn.model_selection import StratifiedKFold
import sys
import os

import torch
torch.set_printoptions(threshold=10_000)

if __name__ == '__main__':

    args = Args()
    args = args.update_args()

    create_dirs(args)

    random.seed(123)

    # graphs = create_graphs(args)

    # random.shuffle(graphs)
    # graphs_train = graphs[: int(0.80 * len(graphs))]
    # graphs_validate = graphs[int(0.80 * len(graphs)): int(0.90 * len(graphs))]

    graph_list = create_graphs(args)

    with open(os.path.join(args.dataset_path, args.graph_type, 'graph_label.dat'), 'rb') as f:
        graph_label_list = pickle.load(f)

    # print('graph_list: {}'.format(graph_list))
    # print('graph_label_list: {}'.format(graph_label_list))

    fold=5
    stratified_KFold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=None)

    graphs_train = []
    graphs_test = []
    for train_index, test_index in stratified_KFold.split(graph_list, graph_label_list):
        graphs_train.append([(graph_list[i], graph_label_list[i]) for i in train_index])
        graphs_test.append([(graph_list[i], graph_label_list[i]) for i in test_index])

    graphs_validate = None

    # print('graphs_train: {}'.format(graphs_train))
    print('graphs_test: {}'.format(graphs_test))

    # show graphs statistics
    print('Model:', args.note)
    print('Device:', args.device)
    print('Graph type:', args.graph_type)
    # print('Training set: {}, Validation set: {}'.format(
    #     len(graphs_train), len(graphs_validate)))
    print('Training set: {}'.format(
        len(graphs_train[0])))

    # Loading the feature map
    with open(args.current_dataset_path + 'map.dict', 'rb') as f:
        feature_map = pickle.load(f)

    if args.note == 'DFScodeRNN_cls':
        feature_map['label_size'] = len(set(graph_label_list))
        print('Number of labels: {}'.format(feature_map['label_size']))

    print('Max number of nodes: {}'.format(feature_map['max_nodes']))
    print('Max number of edges: {}'.format(feature_map['max_edges']))
    print('Min number of nodes: {}'.format(feature_map['min_nodes']))
    print('Min number of edges: {}'.format(feature_map['min_edges']))
    print('Max degree of a node: {}'.format(feature_map['max_degree']))
    print('No. of node labels: {}'.format(len(feature_map['node_forward'])))
    print('No. of edge labels: {}'.format(len(feature_map['edge_forward'])))

    # sys.exit()

    # Setting max_prev_node / max_tail_node and max_head_node
    # yuhao, didnot use this network
    # if args.note == 'GraphRNN':
    #     start = time.time()
    #     if args.max_prev_node is None:
    #         args.max_prev_node = calc_max_prev_node(
    #             args.current_processed_dataset_path)

    #     args.max_head_and_tail = None
    #     print('max_prev_node:', args.max_prev_node)

    #     end = time.time()
    #     print('Time taken to calculate max_prev_node = {:.3f}s'.format(
    #         end - start))

    # yuhao, default is false
    # if args.note == 'DFScodeRNN' and args.weights:
    #     feature_map = {
    #         **feature_map,
    #         **dfscodes_weights(args.min_dfscode_path, graphs_train, feature_map, args.device)
    #     }

    if args.note == 'GraphRNN':
        random_bfs = True
        dataset_train = Graph_Adj_Matrix_from_file(
            args, graphs_train, feature_map, random_bfs)
        dataset_validate = Graph_Adj_Matrix_from_file(
            args, graphs_validate, feature_map, random_bfs)
    elif args.note == 'DFScodeRNN':
        dataset_train = Graph_DFS_code_from_file(
            args, graphs_train, feature_map)
        dataset_validate = Graph_DFS_code_from_file(
            args, graphs_validate, feature_map)
    elif args.note == 'DFScodeRNN_cls':
        dataset_train = Graph_DFS_code_from_file(
            args, graphs_train[0], feature_map)
    elif args.note == 'DGMG':
        dataset_train = DGMG_Dataset_from_file(args, graphs_train, feature_map)
        dataset_validate = DGMG_Dataset_from_file(
            args, graphs_validate, feature_map)

    if args.note == 'DGMG':
        dataloader_train = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.num_workers, collate_fn=dataset_train.collate_batch)
        dataloader_validate = DataLoader(
            dataset_validate, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=dataset_validate.collate_batch)
    elif args.note == 'DFScodeRNN_cls':
        dataloader_train = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.num_workers)
    else:
        dataloader_train = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.num_workers)
        dataloader_validate = DataLoader(
            dataset_validate, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)

    model = create_model(args, feature_map)

    # train(args, dataloader_train, model, feature_map, dataloader_validate)
    train(args, dataloader_train, model, feature_map)
