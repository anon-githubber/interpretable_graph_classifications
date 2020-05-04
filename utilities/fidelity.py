import torch
import networkx as nx
import torch.nn.functional as F
from utilities.util import graph_to_tensor
from statistics import stdev
import random
from copy import deepcopy


def get_accuracy(classified_classes, label):
    count = 0
    for classified_class in classified_classes:
        if classified_class == label:
            count += 1
    return count/len(classified_classes)


def get_class_prob(graph_list, classifier, dataset_features, use_gpu):
    node_feat, n2n_sp, subg_sp = graph_to_tensor(
        graph_list, dataset_features["feat_dim"],
        dataset_features["edge_feat_dim"], use_gpu)
    output = classifier(node_feat, n2n_sp, subg_sp, graph_list)
    logits = F.log_softmax(output, dim=1)
    pred = logits.data.max(1, keepdim=True)[1]
    pred = pred.tolist()
    pred = [x[0] for x in pred]
    return pred


def is_important(score, importance_ranges):
    for start, end in importance_ranges:
        if start <= score <= end:
            return True
    return False


def get_fidelity_graph(GNNgraph_list, scores_list, importance_ranges):
    """
    Returns a new graph after occluding the salient nodes

    :param GNNgraph_list: A list of GNNGraph objects
    :param scores_list: Attribution scores of nodes from each GNNGraph
    :param importance_ranges: A list of sets containing the range(s) from which salient nodes are determined.
                              To define more than 1 ranges, append a new set to the list. Ranges are inclusive on both sides.
    :return: returns a list of modified GNNGraph 
    """
    graph_list = []
    # transform the graphs, occlude nodes with positive attribution score
    for GNNgraph, attribution_scores in zip(GNNgraph_list, scores_list):
        attribution_scores_list = []

        for score in attribution_scores:
            attribution_scores_list.append(score.item())

        unk_index = max(GNNgraph.node_tags_dict.keys()) + 1
        GNNgraph.node_tags_dict[unk_index] = 'UNK'

        for idx in range(len(attribution_scores)):
            if is_important(attribution_scores[idx], importance_ranges):
                GNNgraph.node_tags[idx] = unk_index
        graph_list.append(GNNgraph)
    return graph_list


def measure_fidelity(classifier_model, sampled_graphs, node_scores_list, dataset_features, importance_ranges=[(0.5, 1)], use_gpu=0):
    fidelities = []
    for label, graph_list in sampled_graphs.items():
        # measure initial accuracy
        classified_classes = get_class_prob(
            deepcopy(graph_list), classifier_model, dataset_features, use_gpu) 
        initial_accuracy = get_accuracy(classified_classes, label)

        # occlude salient nodes
        fidelity_graphs = get_fidelity_graph(
            deepcopy(graph_list), node_scores_list, importance_ranges)

        # measure final accuracy
        classified_classes = get_class_prob(
            deepcopy(fidelity_graphs), classifier_model, dataset_features, use_gpu)
        final_accuracy = get_accuracy(classified_classes, label)
        fidelities.append(
            initial_accuracy-final_accuracy)

    avg_fidelity_score = sum(fidelities)/len(fidelities)
    std_dev = stdev(fidelities)
    return avg_fidelity_score, std_dev
    # print('Fidelity score: %.5f ± %.5f' %
    #       (avg_fidelity_score, std_dev))
