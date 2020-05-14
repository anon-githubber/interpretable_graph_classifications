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
import networkx as nx
import networkx.algorithms.isomorphism as iso
from torch.utils.data.sampler import SubsetRandomSampler

from utilities.load_data import load_model_data, unserialize_pickle_file
from utilities.util import graph_to_tensor, get_node_labels_dict
from utilities.output_results import output_to_images, output_subgraph_images, output_subgraph_list_to_images
from utilities.metrics import auc_scores, is_salient
from utilities.graphsig import convert_graphsig_to_gnn_graph

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
	cmd_opt.add_argument('-graphsig', default='0',
						 help='Perform graphsig subgraph analysis if 1')
	cmd_opt.add_argument('-subgraph_explainability', default='0',
						 help='Perform explainability subgraph analysis if 1')
	cmd_opt.add_argument('-graphsig_classification', default='0',
						 help='Perform graphsig classification if 1')
	cmd_args, _ = cmd_opt.parse_known_args()

	# Get run configurations
	config = yaml.safe_load(open("config.yml"))

	# Set random seed
	random.seed(config["run"]["seed"])
	np.random.seed(config["run"]["seed"])
	torch.manual_seed(config["run"]["seed"])

	# Load graph data using util.load_data(), see util.py ==============================================================
	# Specify the dataset to use and the number of folds for partitioning
	train_graphs, test_graphs, dataset_features = load_model_data(
		cmd_args.data,
		config["run"]["k_fold"],
		config["general"]["data_autobalance"],
		config["general"]["print_dataset_features"]
	)

	print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))
	config["dataset_features"] = dataset_features

	# Instantiate the classifier using the configurations ==============================================================
	# Use saved model only for subgraph analysis
	if cmd_args.retrain == '0' and cmd_args.subgraph_explainability == '1':
		# Load classifier if it exists:
		model_list = None
		try:
			model_list = torch.load("tmp/saved_models/%s_%s_epochs_%s_learnrate_%s_folds_%s.pth" %
									(dataset_features["name"], cmd_args.gm, str(config["run"]["num_epochs"]),
									 str(config["run"]["learning_rate"]), str(config["run"]["k_fold"])))
		except FileNotFoundError:
			print("Retrain is disabled but no such save of %s for dataset %s with the current training configurations"
				  " exists in tmp/saved_models folder. "
				  "Please retry run with -retrain enabled." % (dataset_features["name"], cmd_args.gm))
			exit()

		print("Testing models using saved model: " + cmd_args.gm)

		for fold_number in range(len(model_list)):
			print("Testing using fold %s" % fold_number)
			model_list[fold_number].eval()

			test_graph_fold = test_graphs[fold_number]

			test_idxes = list(range(len(test_graph_fold)))
			test_loss = loop_dataset(test_graph_fold, model_list[fold_number], test_idxes,
									 config, dataset_features)
			print('\033[93maverage test: loss %.5f acc %.5f roc_auc %.5f prc_auc %.5f\033[0m' % (
				test_loss[0], test_loss[1], test_loss[2], test_loss[3]))
	elif cmd_args.retrain != '0':
		print("Please use saved model to perform subgraph analysis.")

	graph_list = deepcopy(train_graphs[0] + test_graphs[0])
	minSup = len(graph_list) // 10

	# Begin performing interpretability methods ========================================================================
	interpretability_methods_config = config["interpretability_methods"]
	start_image = time.perf_counter()
	if cmd_args.subgraph_explainability == '1':
		for method in config["interpretability_methods"].keys():
			if config["interpretability_methods"][method]["enabled"] is False:
				continue

			print("Running method: " + str(method))
			exec_string = "score_output, saliency_output, generate_score_execution_time = " \
				"%s(model_list[0], config," \
				" dataset_features, test_graphs[fold_number], fold_number, cmd_args.cuda)" % method
			exec(exec_string)

			if cmd_args.subgraph_explainability == '1':
				# Get significant subgraphs from output =========================================================================
				# Remove irrelevant nodes
				importance_range = config["metrics"]["fidelity"]["importance_range"].split(
					",")
				importance_range = [float(bound) for bound in importance_range]

				modified_graphs = {0: [], 1: []}
				for data in score_output:
					graph = data['graph']
					label = graph.label
					class_0_score = data[0]
					class_1_score = data[1]
					graph = graph.to_nxgraph()
					nodes_to_delete = []
					score_to_use = class_0_score if label == 0 else class_1_score
					for idx, node in enumerate(graph.nodes()):
						if not is_salient(score_to_use[idx], importance_range):
							nodes_to_delete.append(node)
					graph.remove_nodes_from(nodes_to_delete)
					modified_graphs[label].append(graph)

				# Generate subgraphs
				subgraphs = {0: [], 1: []}
				for label, sg_list in modified_graphs.items():
					print('Generating subgraph class %s using %s' %
						  (label, method))
					for sg in sg_list:
						component_subgraphs = [sg.subgraph(
							c).copy() for c in nx.connected_components(sg)]
						for sg in component_subgraphs:
							subgraphs[sg.graph['label']].append(sg)
					print('Generated %s class %s subgraphs using %s' % (len(subgraphs[label]), label, method))

				# Calculate the frequencies in sample graphs
				subgraphs_info = {0: [], 1: []}
				for label, subgraph_list in subgraphs.items():
					print('Counting frequencies for %s subgraph class %s' %
						  (method, label))
					for subgraph in subgraph_list:
						class_0_count = 0
						class_1_count = 0
						graph_samples = random.sample(graph_list, 300)
						# graph_samples = graph_list
						minSup = len(graph_samples) // 10
						nm = iso.categorical_node_match('label', '')
						for graph in graph_samples:
							GM = isomorphism.GraphMatcher(
								graph.to_nxgraph(), subgraph, node_match=nm)
							if GM.subgraph_is_isomorphic():
								if graph.label == 0:
									class_0_count += 1
								elif graph.label == 1:
									class_1_count += 1
						subgraphs_info[label].append(
							(subgraph, class_0_count, class_1_count))

				# Remove isomorphic subgraphs
				for label, subgraph_list in subgraphs_info.items():
					print('Removing isomorphic subgraphs for %s class %s' %
						  (method, label))
					new_list = []
					j = 0
					nm = iso.categorical_node_match('label', -1)
					for subgraph in subgraph_list:
						is_there = False
						sgnx = subgraph[0]
						for new_sg in new_list:
							new_sgnx = new_sg[0]
							GM = isomorphism.GraphMatcher(new_sgnx, sgnx, node_match=nm)
							if nx.is_isomorphic(new_sgnx, sgnx, node_match=nm) or GM.subgraph_is_isomorphic():
								is_there = True
								break
						if not is_there:
							new_list.append(subgraph)
					subgraphs_info[label] = new_list
					print('After removing isomorphic subgraphs, class %s left with %s graphs' % (label, len(new_list)))

				# Filter by min sup and sort by frequencies
				for label, subgraphs_list in subgraphs_info.items():
					print('Sorting and filtering %s subgraph for class %s' %
						  (method, label))
					subgraphs_list = list(
						filter(lambda x: x[label + 1] >= minSup, subgraphs_list))
					subgraphs_info[label] = subgraphs_list
					subgraphs_list.sort(key=lambda x: len(
						x[0].nodes()), reverse=True)
					print('After filtering, class %s left with %s graphs' % (label, len(subgraphs_list)))

				# Output top 5 to image
				node_labels_dict = get_node_labels_dict(cmd_args.data)
				for label, subgraphs_list in subgraphs_info.items():
					output_subgraph_list_to_images(
						subgraphs_list, dataset_features, method, label, node_labels_dict, has_frequency=True)
					print('Subgraph images (%s) for %s class %s saved' %
						  (cmd_args.data, method, label))

	if cmd_args.graphsig == '1':
		# GraphSig subgraph analysis
		# Load GraphSig significant subgraphs
		graphsig_subgraph_list_class_0 = unserialize_pickle_file(
			'data/%s/%s_class_0_graphsig' % (cmd_args.data, cmd_args.data))
		graphsig_subgraph_list_class_1 = unserialize_pickle_file(
			'data/%s/%s_class_1_graphsig' % (cmd_args.data, cmd_args.data))
		graphsig_subgraphs = {0: graphsig_subgraph_list_class_0,
							  1: graphsig_subgraph_list_class_1}

		node_labels_dict = get_node_labels_dict(cmd_args.data)

		print('No of subgraphs:')
		print('Class 0: %d' % len(graphsig_subgraphs[0]))
		print('Class 1: %d' % len(graphsig_subgraphs[1]))

		# # Save subgraphs images
		# for label, graphsig_subgraph in graphsig_subgraphs.items():
		# 	output_subgraph_list_to_images(graphsig_subgraph, dataset_features, 'GraphSig', label, node_labels_dict, print_rank=False)
		# 	print('GraphSig subgraphs for class %s saved' % label)

		# Get frequencies for significant subgraphs from GraphSig in randomly sampled graphs
		graphsig_subgraphs_info = {0: [], 1: []}
		for label, subgraph_list in graphsig_subgraphs.items():
			print('Counting GraphSig subgraph frequencies for label %s' % label)
			with(open('data/%s/graphsig_frequencies_label_%d' % (cmd_args.data, label), 'w')) as out_file:
				for subgraph in subgraph_list:
					subgraphnx = subgraph.to_nxgraph()
					class_0_count = 0
					class_1_count = 0
					# sample = graph_list
					random.seed(1800)
					sample = random.sample(graph_list, 300)
					nm = iso.categorical_node_match('label', -1)
					for graph in sample:
						graphnx = graph.to_nxgraph()
						GM = isomorphism.GraphMatcher(
							graphnx, subgraphnx, node_match=nm)
						if GM.subgraph_is_isomorphic():
							if graph.label == 0:
								class_0_count += 1
							elif graph.label == 1:
								class_1_count += 1
					graphsig_subgraphs_info[label].append(
						(subgraph, class_0_count, class_1_count))
					print(str(class_0_count) + ' ' + str(class_1_count), file=out_file)

		# Remove isomorphic subgraphs
		for label, subgraph_list in graphsig_subgraphs_info.items():
			print('Removing isomorphic subgraphs for %s class %s' %
				  ('GraphSig', label))
			new_list = []
			j = 0
			nm = iso.categorical_node_match('label', -1)
			for subgraph in subgraph_list:
				is_there = False
				sg_nx = subgraph[0].to_nxgraph()
				for new_sg in new_list:
					new_sg_nx = new_sg[0].to_nxgraph()
					GM = isomorphism.GraphMatcher(new_sg_nx, sg_nx, node_match=nm)
					if nx.is_isomorphic(new_sg_nx, sg_nx, node_match=nm) or GM.subgraph_is_isomorphic():
						is_there = True
						break
				if not is_there:
					new_list.append(subgraph)
			graphsig_subgraphs_info[label] = new_list
			print('No of class %d subgraph after removing isomorphic graphs: %d' % (label, len(new_list)))

		# Sort by frequencies
		for label, subgraphs_list in graphsig_subgraphs_info.items():
			print('Sorting GraphSig subgraph (%s) data for label %s' %
				  (cmd_args.data, label))
			subgraphs_list.sort(key=lambda x: x[label + 1], reverse=True)

		# Output top 10 to image
		for label, subgraphs_list in graphsig_subgraphs_info.items():
			print('Saving GraphSig subgraph (%s) data for label %s to images' %
				  (cmd_args.data, label))
			output_subgraph_list_to_images(
				[(sg[0].to_nxgraph(), sg[1], sg[2]) for sg in subgraphs_list[:10]], dataset_features, 'GraphSig', label, node_labels_dict, has_frequency=True)

	if cmd_args.graphsig_classification == '1':
		# GraphSig classification analysis
		# Load GraphSig significant subgraphs
		graphsig_subgraph_list_class_0 = unserialize_pickle_file(
			'data/%s/%s_class_0_graphsig' % (cmd_args.data, cmd_args.data))
		graphsig_subgraph_list_class_1 = unserialize_pickle_file(
			'data/%s/%s_class_1_graphsig' % (cmd_args.data, cmd_args.data))
		for graphsig_sg in graphsig_subgraph_list_class_1:
			graphsig_sg.label = 1
		graphsig_subgraphs = graphsig_subgraph_list_class_0 + graphsig_subgraph_list_class_1
		dataset = train_graphs[0] + test_graphs[0]
		graphsig_subgraphs = random.sample(graphsig_subgraphs, 200)
		print('Number of subgraphs: %d' % len(graphsig_subgraphs))
		# Load 1D array to tensor if it's saved previously
		start_conversion = time.perf_counter()
		fname = 'data/%s/1d_vector.txt' % cmd_args.data
		try:
			f = open(fname, 'r')
		except OSError:
			print("Could not open/read file:", fname)
			print("Saving a new file")
			# Convert dataset to 1D array and save if doesn't exist
			print('Converting data to 1D array....')
			converted_data = []
			labels = []
			for data in dataset:
				vector = ''
				labels.append(data.label)
				for subgraph in graphsig_subgraphs:
					nm = iso.categorical_node_match('label', -1)
					GM = isomorphism.GraphMatcher(
						data.to_nxgraph(), subgraph.to_nxgraph(), node_match=nm)
					if GM.subgraph_is_isomorphic():
						vector += '1'
					else:
						vector += '0'
				converted_data.append(vector)

			with open('data/%s/1d_vector.txt' % cmd_args.data, 'w') as output_file:
				for data in converted_data:
					print(data, file=output_file)
				print("1D Dataset for %s saved" % cmd_args.data)
				output_file.close()

			with open('data/%s/1d_vector_labels.txt' % cmd_args.data, 'w') as output_file:
				for data in labels:
					print(data, file=output_file)
				print("1D labels for %s saved" % cmd_args.data)
				output_file.close()

			total_conversion_time = time.perf_counter() - start_conversion
			print('Total time taken for conversion (s): %s' %
				  str(total_conversion_time))

		f = open(fname, 'r')
		tensors = []
		with f:
			temp = f.read().splitlines()
			for t in temp:
				arr = [int(x) for x in list(t)]
				tensors.append(torch.tensor(arr).float())

		labelname = 'data/%s/1d_vector_labels.txt' % cmd_args.data
		f = open(labelname, 'r')
		labels = []
		with f:
			temp = f.read().splitlines()
			for t in temp:
				labels.append(int(t))

		net = DNN(input_dim=len(tensors[0]))
		dataset = []
		for i in range(len(tensors)):
			dataset.append((tensors[i], int(labels[i])))

		# create a stochastic gradient descent optimizer
		optimizer = optim.SGD(
			net.parameters(), lr=config['run']['learning_rate'])
		# create a loss function
		criterion = nn.CrossEntropyLoss()

		# Creating data indices for training and validation splits:
		validation_split = .2
		dataset_size = len(dataset)
		indices = list(range(dataset_size))
		split = int(np.floor(validation_split * dataset_size))
		shuffle_dataset = True
		if shuffle_dataset:
			np.random.seed(config['run']['seed'])
			np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]

		# Creating PT data samplers and loaders:
		train_sampler = SubsetRandomSampler(train_indices)
		valid_sampler = SubsetRandomSampler(val_indices)

		train_loader = torch.utils.data.DataLoader(dataset, batch_size=config['general']['batch_size'],
												   sampler=train_sampler)
		validation_loader = torch.utils.data.DataLoader(dataset, batch_size=config['general']['batch_size'],
														sampler=valid_sampler)

		# Training and testing
		temp_timing_dict = {'forward': [], 'backward': []}
		log_interval = 10
		all_targets = []
		all_scores = []
		for epoch in range(config['run']['num_epochs']):
			# Train:
			for batch_idx, (data, target) in enumerate(train_loader):
				data, target = data.float(), target.long()
				optimizer.zero_grad()
				start_forward = time.perf_counter()
				net_out = net(data)
				temp_timing_dict["forward"].append(time.perf_counter() - start_forward)
				loss = criterion(net_out, target)
				start_backward = time.perf_counter()
				loss.backward()
				temp_timing_dict["backward"].append(time.perf_counter() - start_backward)
				optimizer.step()
				prob = F.softmax(net_out, dim=1)
				all_scores.append(prob.cpu().detach())
				all_targets.append(target)
			roc_auc, prc_auc = auc_scores(all_targets, all_scores)
			if (epoch + 1) % log_interval == 0:
				print('Train Epoch: {} --- \tLoss: {:.6f}\tROC_AUC: {:.6f} \tPRC_AUC: {:.6f}'.format(
					epoch + 1, loss.item(), roc_auc, prc_auc))

			# Test:
			for batch_idx, (data, target) in enumerate(validation_loader):
				data, target = data.float(), target.long()
				optimizer.zero_grad()
				start_forward = time.perf_counter()
				net_out = net(data)
				temp_timing_dict["forward"].append(time.perf_counter() - start_forward)
				loss = criterion(net_out, target)
				start_backward = time.perf_counter()
				loss.backward()
				temp_timing_dict["backward"].append(time.perf_counter() - start_backward)
				optimizer.step()
				prob = F.softmax(net_out, dim=1)
				all_scores.append(prob.cpu().detach())
				all_targets.append(target)
			roc_auc, prc_auc = auc_scores(all_targets, all_scores)
			if (epoch + 1) % log_interval == 0:
				print('Test Epoch: {} --- \tLoss: {:.6f}\tROC_AUC: {:.6f} \tPRC_AUC: {:.6f}'.format(
					epoch + 1, loss.item(), roc_auc, prc_auc))

		# Print time taken
		print('Time taken for GraphSig classification: ')
		graphsig_time = {}
		graphsig_time['forward'] = sum(temp_timing_dict['forward'])/len(temp_timing_dict['forward'])
		graphsig_time['backward'] = sum(temp_timing_dict['backward'])/len(temp_timing_dict['backward'])
		print('Forward pass: %f' % graphsig_time['forward'])
		print('Backward pass: %f' % graphsig_time['backward'])