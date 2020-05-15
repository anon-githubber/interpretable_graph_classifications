from __future__ import print_function
import numpy as np
import random
import torch
import json
import os
import networkx as nx
from utilities.GNNGraph import GNNGraph
import matplotlib.pyplot as plt
import matplotlib


def output_to_images(output, dataset_features, custom_label_mapping=None, output_directory="results/image"):
    '''

    :param output: the output data structure obtained from a interpretability method. It follows the following format:
                    {output_group_1: [(nxgraph_1, attribution_score_list_1) ... (nxgraph_N, attribution_score_list_N)],
                    output_group_2: ...}
    :param dataset_features: a dictionary of useful information about the dataset, obtained from load_data.py
    :param output_path: the path to output the image files
    '''

    group_count = 0
    total_group_count = len(output)
    total_output_count = 0

    for attribution_score_group, graph_score_pair in output.items():
        image_count_to_generate = len(graph_score_pair)
        i = 0
        for pair in graph_score_pair:
            print("output_to_images: Generating image [%s/%s] for group %s [%s/%s]" %
                  (str((i+1)), str(image_count_to_generate), str(attribution_score_group),
                   str(group_count+1), str(total_group_count)))
            GNNgraph = pair[0]
            attribution_scores = pair[1]

            # Get nxgraph from GNNgraph
            nxgraph = GNNgraph.to_nxgraph()

            # Obtain and normalise attribution score
            attribution_scores_list = []
            for score in attribution_scores:
                attribution_scores_list.append(score)

            max_abs_value = max(map(abs, attribution_scores_list))

            # Restore node and graph labels to the same as dataset
            inverse_graph_label_dict = {
                v: k for k, v in dataset_features["label_dict"].items()}
            inverse_node_label_dict = {
                v: k for k, v in dataset_features["node_dict"].items()}

            # Obtain original node labels if mapping is available, else leave blank for all nodes
            if dataset_features["have_node_labels"] is True:
                node_labels = {x[0]: inverse_node_label_dict[x[1]]
                               for x in nxgraph.nodes("label")}

            else:
                node_labels = {x[0]: " " for x in nxgraph.nodes("label")}

            # Obtain original graph label
            graph_label = inverse_graph_label_dict[GNNgraph.label]

            # Draw the network graph
            # Get position of nodes using kamada_kawai layout
            pos = nx.kamada_kawai_layout(nxgraph)
            nodes = nxgraph.nodes()
            ec = nx.draw_networkx_edges(nxgraph, pos, alpha=0.2)
            nc = nx.draw_networkx_nodes(nxgraph, pos, nodelist=nodes,
                                        node_color=attribution_scores_list, vmin=-max_abs_value, vmax=max_abs_value,
                                        with_labels=False, node_size=200, cmap=plt.cm.coolwarm)

            nt = nx.draw_networkx_labels(
                nxgraph, pos, node_labels, font_size=12)

            plt.title("%s ID:%s Label:%s Index:%s" % (
                attribution_score_group, GNNgraph.graph_id, graph_label, str(i)))

            plt.axis('off')
            plt.colorbar(nc)

            # Output image to file
            directory_name = output_directory + "/" + dataset_features["name"]
            try:
                # Create target Directory if not exist
                os.mkdir(directory_name)
                print("Directory ", directory_name,
                      " created in results directory")
            except FileExistsError:
                pass

            image_output_path = "%s/%s/%s_index_%s.png" % (
                output_directory, dataset_features["name"], str(attribution_score_group), str(i))
            plt.savefig(image_output_path)
            plt.clf()
            i += 1
        group_count += 1
        total_output_count += i
    return total_output_count


def output_subgraph_images(subgraph_info_list, dataset_features, method, print_rank=True, output_path="results/subgraph_analysis"):
    '''
    :param subgraph_info: information of subgraph frequencies in the following form
                                                                      [(subgraph,  class_0_frequency, class_1_frequency)]. Input should be sorted by the actual class frequency.
    :param output_path: the path to output the image files
    '''
    for idx, subgraph_info in enumerate(subgraph_info_list):
        nxgraph, class_0_frequency, class_1_frequency = subgraph_info

        # Restore node and graph labels to the same as dataset
        inverse_graph_label_dict = {v: k for k,
                                    v in dataset_features["label_dict"].items()}
        inverse_node_label_dict = {v: k for k,
                                   v in dataset_features["node_dict"].items()}

        # Obtain original node labels if mapping is available, else leave blank for all nodes
        if dataset_features["have_node_labels"] is True:
            node_labels = {x[0]: inverse_node_label_dict[x[1]]
                           for x in nxgraph.nodes("label")}

        else:
            node_labels = {x[0]: " " for x in nxgraph.nodes("label")}

        # Obtain original graph label
        graph_label = inverse_graph_label_dict[nxgraph.graph['label']]

        # Draw the network graph
        # Get position of nodes using kamada_kawai layout
        pos = nx.kamada_kawai_layout(nxgraph)
        nodes = nxgraph.nodes()
        ec = nx.draw_networkx_edges(nxgraph, pos, alpha=0.2)
        nc = nx.draw_networkx_nodes(
            nxgraph, pos, nodelist=nodes, with_labels=False, node_size=200)

        nt = nx.draw_networkx_labels(
            nxgraph, pos, node_labels, font_size=15)

        class_0_freq = "Class 0 frequency: %d" % class_0_frequency
        class_1_freq = "Class 1 frequency: %d" % class_1_frequency
        if print_rank:
            title = "Frequent subgraph, Label:%s \n %s\n%s" % (
                graph_label, idx + 1, class_0_freq, class_1_freq)
        else:
            title = "Frequent subgraph, Label:%s \n %s\n%s" % (
                graph_label, class_0_freq, class_1_freq)
        plt.title(title, fontdict={'fontsize': 8, 'fontweight': 'medium'})
        plt.axis('off')

        # Output image to file
        directory_name = output_path + "/" + \
            dataset_features["name"] + "/" + \
            method + "/class_" + str(graph_label)
        try:
            # Create target Directory if not exist
            os.mkdir(directory_name)
            print("Directory ", directory_name,
                  " created in results directory")
        except FileExistsError:
            pass

        image_output_path = "%s/%s/%s/class_%s/subgraph_rank_%d.png" % (
            output_path, dataset_features["name"], method, graph_label, idx)
        plt.savefig(image_output_path)
        plt.clf()


def output_subgraph_list_to_images(subgraph_list, dataset_features, method, label, node_labels_dict, has_frequency=False, print_rank=True, output_path="results/subgraph_analysis_large"):
    '''
    :param subgraph_list: a list of GNNGraph
    :param output_path: the path to output the image files
    '''
    for idx, subgraph in enumerate(subgraph_list):

        if has_frequency:
            nxgraph, class_0_frequency, class_1_frequency = subgraph
        else:
            nxgraph = subgraph.to_nxgraph()

        # Restore node and graph labels to the same as dataset
        inverse_graph_label_dict = {v: k for k,
                                    v in dataset_features["label_dict"].items()}
        inverse_node_label_dict = {v: k for k,
                                   v in dataset_features["node_dict"].items()}

        # Obtain original node labels if mapping is available, else leave blank for all nodes
        if dataset_features["have_node_labels"] is True:
            node_labels = {x[0]: node_labels_dict.get(inverse_node_label_dict[x[1]])
                           for x in nxgraph.nodes("label")}
        else:
            node_labels = {x[0]: " " for x in nxgraph.nodes("label")}

        # Draw the network graph
        # Get position of nodes using kamada_kawai layout
        pos = nx.kamada_kawai_layout(nxgraph)
        nodes = nxgraph.nodes()
        ec = nx.draw_networkx_edges(nxgraph, pos, alpha=0.2)
        nc = nx.draw_networkx_nodes(
            nxgraph, pos, nodelist=nodes, with_labels=False, node_size=300, node_color='white', edgecolors='black')

        nt = nx.draw_networkx_labels(
            nxgraph, pos, node_labels, font_size=12)

        if print_rank:
            title = "Frequent subgraph, Label:%s \n Rank: %d" % (
                label, idx + 1)
        else:
            title = "Frequent subgraph, Label:%s" % (
                label)

        if has_frequency:
            title += "\n Class 0 freq: %d \n Class 1 freq: %d" % (class_0_frequency, class_1_frequency)
        plt.title(title, fontdict={'fontsize': 12, 'fontweight': 'medium', 'verticalalignment':'top'}, x=0.8, y=0.2)
        plt.axis('off')

        # Output image to file
        directory_name = output_path + "/" + \
            dataset_features["name"] + "/" + \
            method + "/class_" + str(label)
        try:
            # Create target Directory if not exist
            os.mkdir(directory_name)
            print("Directory ", directory_name,
                  " created in results directory")
        except FileExistsError:
            pass

        image_output_path = "%s/%s/%s/class_%s/subgraph_idx_%d.png" % (
            output_path, dataset_features["name"], method, label, idx)
        plt.savefig(image_output_path)
        plt.clf()
