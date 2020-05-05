import torch
import networkx as nx
import json

from os import path
from captum.attr import DeepLift
from utilities.util import graph_to_tensor, normalize_scores

def get_isomorphic_pairs(dataset_name, graph_list, max_pairs=5):
	'''
		Get isomorphic pairs to serve as a baseline to be used in DeepLIFT
	:param data_file_name: name of the dataset
	:param graph_list: a list of graphs to obtain isomorphic pairs
	:param max_pairs: max number of pairs to find. Set this to be low to decrease execution time
	:return: two sets of list that contain indices of the isomorphic graphs
	'''

	# Check if temporary file storing isomorphic pairs exist.
	# This is used to reduce time for repeated experiments
	if path.exists("tmp/deeplift/isopairs_" + dataset_name + ".json"):
		with open("tmp/deeplift/isopairs_" + dataset_name + ".json", 'r') as f:
			indexes = json.load(f)

		class_0_indices = indexes[0]
		class_1_indices = indexes[1]

	else:
	# If no such file exist, then run loop to find isomorphic pairs
		# Split input graph set by class
		i = 0
		iso_graph_indices = [[], []]
		for GNNgraph in graph_list:
			iso_graph_indices[GNNgraph.label].append(i)
			i += 1

		# Function currently only supports binary class labels
		if len(iso_graph_indices) != 2:
			print("ERROR: Only binary graph labels are supported for obtaining isomorphic pairs")
			exit()

		# Run loop to find isomorphic graphs. Exit when max pairs are found
		print("Finding isomorphic graphs. This may take awhile.")
		class_0_indices = []
		class_1_indices = []
		pairs_found = 0
		max_pairs_reached = False
		for GNNgraph_0_index in iso_graph_indices[0]:
			if max_pairs_reached:
				break

			for GNNgraph_1_index in iso_graph_indices[1]:
				if nx.is_isomorphic(graph_list[GNNgraph_0_index].to_nxgraph(),
									graph_list[GNNgraph_1_index].to_nxgraph()):
					class_0_indices.append(GNNgraph_0_index)
					class_1_indices.append(GNNgraph_1_index)
					pairs_found += 1

					if pairs_found >= max_pairs:
						max_pairs_reached = True
						break

		with open("tmp/deeplift/isopairs_" + dataset_name + ".json", 'w') as f:
			f.write(json.dumps([class_0_indices, class_1_indices]))

	# Return the graphs based on the index of the isomorphic pairs
	return [graph_list[i] for i in class_0_indices], [graph_list[i] for i in class_1_indices]

def DeepLIFT(classifier_model, config, dataset_features, graph_list_inputs, cuda=0):
	'''
		:param classifier_model: trained classifier model
		:param config: parsed configuration file of config.yml
		:param dataset_features: a dictionary of dataset features obtained from load_data.py
		:param graph_list_inputs: a list of GNNgraphs
		:param graph_list_inputs_label: a list of labels, arranged accordingly to graph_list_inputs
	'''
	# Initialise settings
	config = config
	dataset_features = dataset_features

	# Perform deeplift on the classifier model
	dl = DeepLift(classifier_model)

	output = {}
	inverse_graph_label_dict = {v: k for k, v in dataset_features["label_dict"].items()}

	# Comparison 1: With zero-tensor
	# Get samples for explanation methods
	if config["compare_with_zero_tensor"] is True:
		sampled_graphs = {}

		for label in range(dataset_features["num_class"]):
			class_sampled_graphs = []
			for GNNgraph in graph_list_inputs:
				if int(GNNgraph.label) == int(label):
					class_sampled_graphs.append(GNNgraph)
			sampled_graphs[label] = class_sampled_graphs

		# Get deeplift score from sampled graphs
		for i in range(len(sampled_graphs)):
			attribution_score_list = []
			for graph_sample in sampled_graphs[i]:
				node_feat_sample, n2n, subg = graph_to_tensor(
					[graph_sample], dataset_features["feat_dim"],
					dataset_features["edge_feat_dim"], cuda)

				subg = subg.size()[0]
				attribution = dl.attribute(node_feat_sample,
										   additional_forward_args=(n2n, subg, [graph_sample]),
										   target=label)
				attribution_score = torch.sum(attribution, dim=1)
				attribution_score = normalize_scores(attribution_score, -1, 1)
				attribution_score_list.append((graph_sample, attribution_score))

			output["deeplift_zero_tensor_class_" + str(i)] = attribution_score_list

	# Comparison 2: With isomers
	# Get isomorphic samples for explanation methods
	if config["compare_with_isomorphic_samples"] is True:
		if dataset_features["num_class"] != 2:
			print("DeepLIFT.py: Comparing with isomorphic samples is only possible in binary classification tasks.")
		else:
			class_0_graphs, class_1_graphs = get_isomorphic_pairs(
				dataset_features["name"], graph_list_inputs,
				config["number_of_isomorphic_sample_pairs"])

			attribution_score_list_0 = []
			attribution_score_list_1 = []

			for graph_0, graph_1 in zip(class_0_graphs, class_1_graphs):
				node_feat_0, n2n, subg = graph_to_tensor(
					[graph_0], dataset_features["feat_dim"],
					dataset_features["edge_feat_dim"], cuda)

				node_feat_1, _, _ = graph_to_tensor(
					[graph_1], dataset_features["feat_dim"],
					dataset_features["edge_feat_dim"], cuda)

				subg = subg.size()[0]

				attribution_0 = dl.attribute(node_feat_0,
					additional_forward_args=(n2n, subg, [graph_0]),
					baselines=node_feat_1,
					target=label)

				attribution_1 = dl.attribute(node_feat_1,
					additional_forward_args=(n2n, subg, [graph_1]),
					baselines=node_feat_0,
					target=label)

				attribution_score_0 = torch.sum(attribution_0, dim=1)
				attribution_score_0 = normalize_scores(attribution_score_0, -1, 1)
				attribution_score_1 = torch.sum(attribution_1, dim=1)
				attribution_score_1 = normalize_scores(attribution_score_1, -1, 1)

				attribution_score_list_0.append((graph_0, attribution_score_0))
				attribution_score_list_1.append((graph_1, attribution_score_1))

			output["deeplift_isomorphic_class_0"] = attribution_score_list_0
			output["deeplift_isomorphic_class_1"] = attribution_score_list_1

	return output