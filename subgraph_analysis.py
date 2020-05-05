import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import time
from copy import deepcopy

import argparse
from models import *
from interpretability_methods import *
from networkx.algorithms import isomorphism

from utilities.load_data import load_model_data
from utilities.util import graph_to_tensor
from utilities.output_results import output_to_images
from utilities.metrics import auc_scores
from utilities.graphsig import convert_graphsig_to_gnn_graph
from utilities.contrastivity import is_important

# Define timer list to report running statistics
timing_dict = {"forward": [], "backward": [], "generate_image": []}


def loop_dataset(g_list, classifier, sample_idxes, config, dataset_features, optimizer=None):
	bsize = max(config["general"]["batch_size"], 1)

	total_loss = []
	total_iters = (len(sample_idxes) + (bsize - 1)
				   * (optimizer is None)) // bsize
	pbar = tqdm(range(total_iters), unit='batch')
	all_targets = []
	all_scores = []

	n_samples = 0

	# Create temporary timer dict to store timing data for this loop
	temp_timing_dict = {"forward": [], "backward": []}

	for pos in pbar:
		selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

		batch_graph = [g_list[idx] for idx in selected_idx]
		targets = [g_list[idx].label for idx in selected_idx]
		all_targets += targets

		node_feat, n2n, subg = graph_to_tensor(
			batch_graph, dataset_features["feat_dim"],
			dataset_features["edge_feat_dim"], cmd_args.cuda)

		subg = subg.size()[0]

		# Get Labels
		labels = torch.LongTensor(len(batch_graph))

		for i in range(len(batch_graph)):
			labels[i] = batch_graph[i].label

		if cmd_args.cuda == 1:
			labels = labels.cuda()

		# Perform training
		start_forward = time.perf_counter()
		output = classifier(node_feat, n2n, subg, batch_graph)
		logits = F.log_softmax(output, dim=1)
		prob = F.softmax(logits, dim=1)

		# Calculate accuracy and loss
		loss = F.nll_loss(logits, labels)
		temp_timing_dict["forward"].append(time.perf_counter() - start_forward)
		pred = logits.data.max(1, keepdim=True)[1]
		acc = pred.eq(labels.data.view_as(pred)).cpu(
		).sum().item() / float(labels.size()[0])
		all_scores.append(prob.cpu().detach())  # for classification

		# Back propagation
		if optimizer is not None:
			start_backward = time.perf_counter()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			temp_timing_dict["backward"].append(
				time.perf_counter() - start_backward)

		loss = loss.data.cpu().detach().numpy()
		pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))
		total_loss.append(np.array([loss, acc]) * len(selected_idx))

		n_samples += len(selected_idx)
	if optimizer is None:
		assert n_samples == len(sample_idxes)
	total_loss = np.array(total_loss)
	avg_loss = np.sum(total_loss, 0) / n_samples

	roc_auc, prc_auc = auc_scores(all_targets, all_scores)
	avg_loss = np.concatenate((avg_loss, [roc_auc], [prc_auc]))

	# Append loop average to global timer tracking list. Only for training phase
	if optimizer is not None:
		timing_dict["forward"].append(
			sum(temp_timing_dict["forward"]) / len(temp_timing_dict["forward"]))
		timing_dict["backward"].append(
			sum(temp_timing_dict["backward"]) / len(temp_timing_dict["backward"]))

	return avg_loss


'''
	Main program execution
'''
if __name__ == '__main__':
	# Get run arguments
	cmd_opt = argparse.ArgumentParser(
		description='Argparser for graph classification')
	cmd_opt.add_argument('-cuda', default='0', help='0-CPU, 1-GPU')
	cmd_opt.add_argument('-gm', default='DGCNN', help='GNN model to use')
	cmd_opt.add_argument('-data', default='TOX21', help='Dataset to use')
	cmd_opt.add_argument('-retrain', default='0',
						 help='Whether to re-train the classifier or use saved trained model')
	cmd_args, _ = cmd_opt.parse_known_args()

	# Get run configurations
	config = yaml.safe_load(open("config.yml"))

	# Set random seed
	random.seed(config["test"]["seed"])
	np.random.seed(config["test"]["seed"])
	torch.manual_seed(config["test"]["seed"])

	# Load graph data using util.load_data(), see util.py ==============================================================
	# Specify the dataset to use and the number of folds for partitioning
	train_graphs, test_graphs, dataset_features = load_model_data(
		cmd_args.data,
		config["test"]["k_fold"],
		config["test"]["test_number"],
		config["general"]["data_autobalance"],
		config["general"]["print_dataset_features"]
	)

	print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))
	config["dataset_features"] = dataset_features

	# Instantiate the classifier using the configurations ==============================================================
	# Use saved model only for subgraph analysis
	if cmd_args.retrain == '0':
		# Load classifier if it exists:
		try:
			classifier_model = torch.load("tmp/saved_models/%s_%s_epochs_%s_learnrate_%s.pth" %
										  (dataset_features["name"], cmd_args.gm,
										   str(config["train"]["num_epochs"]), str(config["train"]["learning_rate"])))
		except FileNotFoundError:
			print("Retrain is disabled but no such save of %s for dataset %s exists in tmp/saved_models folder. "
				  "Please Retry run with -retrain enabled." % (dataset_features["name"], cmd_args.gm))
			exit()

		print("Testing model using saved model: " + cmd_args.gm)
		classifier_model.eval()

		test_idxes = list(range(len(test_graphs)))
		test_loss = loop_dataset(test_graphs, classifier_model, test_idxes,
								 config, dataset_features)
		print('\033[93maverage test: loss %.5f acc %.5f roc_auc %.5f prc_auc %.5f\033[0m' % (
			test_loss[0], test_loss[1], test_loss[2], test_loss[3]))
	else:
		print("Please use saved model to perform subgraph analysis.")

	# Group sample graphs by label
	sampled_graphs = {}
	for graph in deepcopy(train_graphs + test_graphs):
		if graph.label in sampled_graphs:
			sampled_graphs[graph.label].append(graph.to_nxgraph())
		else:
			sampled_graphs[graph.label] = [graph.to_nxgraph()]

	# Begin performing interpretability methods ========================================================================
	interpretability_methods_config = config["interpretability_methods"]
	start_image = time.perf_counter()
	for method in config["interpretability_methods"].keys():
		print("Running method: " + str(method))
		exec_string = "output = %s(classifier_model, config[\"interpretability_methods\"][\"%s\"], dataset_features, " \
			"train_graphs + test_graphs, cmd_args.cuda)" % (method, method)
		exec(exec_string)

		# TODO: Get significant subgraphs from output =========================================================================
		# Remove irrelevant nodes
		output_list = output['deeplift_zero_tensor_class_0'] + \
			output['deeplift_zero_tensor_class_1']

		modified_graphs = []
		for graph, score in output_list:
			graph = graph.to_nxgraph()
			nodes_to_delete = []
			for idx, node in enumerate(graph.nodes()):
				if not is_important(score[idx], [(0.5, 1), (-1, -0.5)]):
					nodes_to_delete.append(node)
			graph.remove_nodes_from(nodes_to_delete)
			modified_graphs.append(graph)

		# Generate subgraphs
		subgraphs = []
		for g in modified_graphs:
			component_subgraphs = [g.subgraph(c).copy() for c in nx.connected_components(g)]
			for sg in component_subgraphs:
				subgraphs.append(sg)

		# TODO: Calculate the frequencies in sample graphs

	# GraphSig subgraph analysis
	# Load GraphSig significant subgraphs
	graphsig_subgraph_list_class_0 = convert_graphsig_to_gnn_graph(
		'data/%s/graphsig/%s_class_0/significantGraphs.txt' % (cmd_args.data, cmd_args.data))
	graphsig_subgraph_list_class_1 = convert_graphsig_to_gnn_graph(
		'data/%s/graphsig/%s_class_1/significantGraphs.txt' % (cmd_args.data, cmd_args.data))

	# Get frequencies for significant subgraphs from GraphSig in sample graphs
	class_0_counts = []
	class_1_counts = []
	for subgraph in graphsig_subgraph_list_class_0:
		class_0_counter = 0
		for graph in sampled_graphs['0']:
			GM = isomorphism.GraphMatcher(graph, subgraph.to_nxgraph())
			if GM.subgraph_is_isomorphic():
				class_0_counter += 1
		class_0_counts.append(class_0_counter)

		class_1_counter = 0
		for graph in sampled_graphs['1']:
			GM = isomorphism.GraphMatcher(graph, subgraph.to_nxgraph())
			if GM.subgraph_is_isomorphic():
				class_1_counter += 1
		class_1_counts.append(class_1_counter)

	for subgraph in graphsig_subgraph_list_class_1:
		class_0_counter = 0
		for graph in sampled_graphs['0']:
			GM = isomorphism.GraphMatcher(graph, subgraph.to_nxgraph())
			if GM.subgraph_is_isomorphic():
				class_0_counter += 1
		class_0_counts.append(class_0_counter)

		class_1_counter = 0
		for graph in sampled_graphs['1']:
			GM = isomorphism.GraphMatcher(graph, subgraph.to_nxgraph())
			if GM.subgraph_is_isomorphic():
				class_1_counter += 1
		class_1_counts.append(class_1_counter)

	subgraph_list = graphsig_subgraph_list_class_0 + graphsig_subgraph_list_class_1

	subgraph_list_with_frequencies = []
	for i in range(subgraph_list):
		subgraph_list_with_frequencies.append(
			(subgraph_list[i], class_0_counts[i], class_1_counts[i]))

	# Sort by various frequencies
	absolute_frequency = sorted(
		subgraph_list_with_frequencies, key=lambda subgraph: subgraph[1] + subgraph[2])
	class_0_frequency = sorted(
		subgraph_list_with_frequencies, key=lambda subgraph: subgraph[1])
	class_1_frequency = sorted(
		subgraph_list_with_frequencies, key=lambda subgraph: subgraph[2])

	# TODO: Output to image
