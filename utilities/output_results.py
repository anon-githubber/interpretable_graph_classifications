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

def output_to_images(output, dataset_features, custom_label_mapping = None, output_directory="results/image"):
	'''

	:param output: the output data structure obtained from a interpretability method. It follows the following format:
		{output_group_1: [(nxgraph_1, attribution_score_list_1) ... (nxgraph_N, attribution_score_list_N)],
		output_group_2: ...}
	:param dataset_features: a dictionary of useful information about the dataset, obtained from load_data.py
	:param output_path: the path to output the image files
	'''
	for attribution_score_group, group_content in output.items():
		i = 0
		for group in group_content:
			GNNgraph = group[0]
			attribution_scores = group[1]

			# Get nxgraph from GNNgraph
			nxgraph = GNNgraph.to_nxgraph()

			# Obtain and normalise attribution score
			attribution_scores_list = []
			for score in attribution_scores:
				attribution_scores_list.append(score.item())

			max_abs_value = max(map(abs, attribution_scores_list))

			# Restore node and graph labels to the same as dataset
			inverse_graph_label_dict = {v: k for k, v in dataset_features["label_dict"].items()}
			inverse_node_label_dict = {v: k for k, v in dataset_features["node_dict"].items()}

			node_labels = {x[0]: inverse_node_label_dict[x[1]] for x in nxgraph.nodes("label")}
			graph_label = inverse_graph_label_dict[GNNgraph.label]

			# Draw the network graph
			# Get position of nodes using kamada_kawai layout
			pos = nx.kamada_kawai_layout(nxgraph)
			nodes = nxgraph.nodes()
			ec = nx.draw_networkx_edges(nxgraph, pos, alpha=0.2)
			nc = nx.draw_networkx_nodes(nxgraph, pos, nodelist=nodes,
										node_color=attribution_scores_list, vmin=-max_abs_value, vmax=max_abs_value,
										with_labels=False, node_size=200, cmap=plt.cm.coolwarm)

			nt = nx.draw_networkx_labels(nxgraph, pos, node_labels, font_size=12)

			plt.title("%s_label_%s_index_%s" % (attribution_score_group, graph_label, str(i)))

			plt.axis('off')
			plt.colorbar(nc)

			# Output image to file
			directory_name = output_directory + "/" + dataset_features["name"]
			try:
				# Create target Directory if not exist
				os.mkdir(directory_name)
				print("Directory ", directory_name, " created in results directory")
			except FileExistsError:
				pass

			image_output_path = "%s/%s/%s_index_%s.png" % (output_directory, dataset_features["name"], str(attribution_score_group), str(i))
			plt.savefig(image_output_path)
			plt.clf()
			i += 1