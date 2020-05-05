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

def binarize_scores_list(scores_list, importance_ranges):
    binarized_scores_list = []
    for scores in scores_list:
        s = ''
        for score in scores:
            if is_important(score, importance_ranges):
                s += '1'
            else:
                s += '0'
        binarized_scores_list.append(s)
    return binarized_scores_list


def post_process_deeplift_scores(original, flipped):
    '''
    :param original: original scores as outputted by DeepLift module
    :param flipped: scores as outputted by DeepLift module after the labels are flipped
    :return returns two lists, each containing scores for both classes. Same index refers to the same input graph.
    '''
    class_0_key = 'deeplift_zero_tensor_class_0'
    class_1_key = 'deeplift_zero_tensor_class_1'
    class_0_scores = []
    class_1_scores = []

    # maps class 0 original to class 1 flipped
    for graph, scores in original[class_0_key]:
        class_0_scores.append(scores)

    for graph, scores in flipped[class_1_key]:
        class_1_scores.append(scores)

    # maps class 1 original to class 0 flipped
    for graph, scores in original[class_1_key]:
        class_1_scores.append(scores)

    for graph, scores in flipped[class_0_key]:
        class_0_scores.append(scores)

    return class_0_scores, class_1_scores


def measure_contrastivity(classifier_model, sampled_graphs, method, dataset_features, config, importance_ranges=[(0.5, 1)], use_gpu=0):
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

    class_0_scores, class_1_scores = post_process_deeplift_scores(
        original_scores, flipped_scores)

    class_0_binarized_scores_list = binarize_scores_list(class_0_scores, importance_ranges)
    class_1_binarized_scores_list = binarize_scores_list(class_1_scores, importance_ranges)

    # measure the average hamming distance across all isomers
    result = []
    for class_0, class_1 in zip(class_0_binarized_scores_list, class_1_binarized_scores_list):
        assert len(class_0) == len(class_1)
        d = hamming(class_0, class_1)
        result.append(d / len(class_0))
    return sum(result)/len(result), stdev(result)
