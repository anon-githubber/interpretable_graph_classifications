import torch
import networkx as nx
import torch.nn.functional as F
from utilities.util import graph_to_tensor, hamming, normalize_scores
from statistics import stdev
import random
from interpretability_methods import *


def is_important(score, importance_ranges):
    for start, end in importance_ranges:
        if start <= score <= end:
            return True
    return False


def get_significant_nodes_count(scores_list, importance_ranges):
    res = []
    for scores in scores_list:
        count = 0
        for score in scores:
            if is_important(score, importance_ranges):
                count += 1
        res.append(count)
    return res


def post_process_deeplift_scores(original, flipped):
    '''
    :param original: original scores as outputted by DeepLift module
    :param flipped: scores as outputted by DeepLift module after the labels are flipped
    :return returns three lists, one containing the graphs, two containing scores for both classes. Same index refers to the same input graph.
    '''
    class_0_key = 'deeplift_zero_tensor_class_0'
    class_1_key = 'deeplift_zero_tensor_class_1'
    class_0_scores = []
    class_1_scores = []
    graph_list = []

    # maps class 0 original to class 1 flipped
    for graph, scores in original[class_0_key]:
        graph_list.append(graph)
        class_0_scores.append(scores)

    for graph, scores in flipped[class_1_key]:
        class_1_scores.append(scores)

    # maps class 1 original to class 0 flipped
    for graph, scores in original[class_1_key]:
        graph_list.append(graph)
        class_1_scores.append(scores)

    for graph, scores in flipped[class_0_key]:
        class_0_scores.append(scores)

    return class_0_scores, class_1_scores, graph_list

def get_graphs_number_of_nodes(graph_list):
    res = []
    for graph in graph_list:
        res.append(graph.number_of_nodes)
    return res

def measure_sparsity(classifier_model, sampled_graphs, method, dataset_features, config, importance_ranges=[(0.5, 1)], use_gpu=0):
    # get attribution scores for original class
    if method == 'DeepLIFT':
        original_scores = DeepLIFT(
            classifier_model, config["interpretability_methods"][method], dataset_features, sampled_graphs, use_gpu)

    # flipped labels
    for graph in sampled_graphs:
        graph.label = 0 if graph.label == 1 else 1

    # get attribution scores for flipped class
    if method == 'DeepLIFT':
        flipped_scores = DeepLIFT(
            classifier_model, config["interpretability_methods"][method], dataset_features, sampled_graphs, use_gpu)

    class_0_scores, class_1_scores, graph_list = post_process_deeplift_scores(
        original_scores, flipped_scores)

    class_0_significant_nodes_count = get_significant_nodes_count(
        class_0_scores, importance_ranges)
    class_1_significant_nodes_count = get_significant_nodes_count(
        class_1_scores, importance_ranges)
    graphs_number_of_nodes = get_graphs_number_of_nodes(graph_list)

    # measure the average sparsity score across all samples
    result = []
    for i in range(len(class_0_significant_nodes_count)):
        d = class_0_significant_nodes_count[i] + \
            class_1_significant_nodes_count[i]
        d /= (graphs_number_of_nodes[i] * 2)
        result.append(1 - d)
    return sum(result)/len(result), stdev(result)