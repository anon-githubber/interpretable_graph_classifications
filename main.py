import pprint

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import time
import datetime
import argparse
import sys

import yaml
import json
import hashlib

from tqdm import tqdm
from copy import deepcopy

# Import user-defined models and interpretability methods
from models import *
#from interpretability_methods import *

# Import user-defined functions
from utilities.load_data import load_model_data, split_train_test
from utilities.util import graph_to_tensor
from utilities.output_results import output_to_images
from utilities.metrics import auc_scores, compute_metric

# Check if gpu is available
print('\n\ntorch.cuda.is_available(): ', torch.cuda.is_available(), '\n\n')


# Define timer list to report running statistics
timing_dict = {"forward": [], "backward": []}
run_statistics_string = "Run statistics: \n"

def loop_dataset_DFSRNN(sample_idxes, graph_label_list, classifier, config, args, dataset_features, optimizer=None):
	'''
	:param g_list: list of graphs to trainover
	:param classifier: the initialised classifier
	:param config: Run configurations as stated in config.yml
	:param dataset_features: Dataset features obtained from load_data.py
	:param optimizer: optimizer to use
	:return: average loss and other model performance metrics
	'''

	from models.graphgen.graphgen_cls.data import load_dfscode_tensor
	from models.graphgen.graphgen_cls.train import get_RNN_input_from_dfscode_tensor

	print('*** 4 len(sample_idxes): ', len(sample_idxes))
	print('*** 5 sample_idxes: ', sample_idxes)
	# print('*** 6 config: ', config)

	n_samples = 0
	all_targets = []
	all_scores = []
	total_loss = []

	# Determine batch size and initialise progress bar (pbar)
	bsize = max(config["general"]["batch_size"], 1)

	total_iters = (len(sample_idxes) + (bsize - 1) *
				   (optimizer is None)) // bsize
	# pbar = tqdm(range(total_iters), unit='batch')
	print(f'*** 6 total_iters: {total_iters}')

	# Create temporary timer dict to store timing data for this loop
	temp_timing_dict = {"forward": [], "backward": []}

	# init classifier
	print(f'*** 7 classifier: {classifier}')
	classifier.dfs_code_rnn.init_hidden(batch_size=bsize)

	# For each batch
	for pos in range(total_iters):
		selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

		batch_graph_tensors = []
		for idx in selected_idx:
			dfscode_tensor = load_dfscode_tensor(args.min_dfscode_tensor_path, idx)
			print(f'*** 7 dfscode_tensor: {dfscode_tensor}')
			
			batch_graph_tensors.append(get_RNN_input_from_dfscode_tensor(dfscode_tensor, bsize, args, dataset_features))
			# print(f'*** 7 batch_graph_tensors: {batch_graph_tensors}')

		batch_graph_tensors = torch.cat(batch_graph_tensors).unsqueeze(0)
		print(f'*** 7 batch_graph_tensors: {batch_graph_tensors}')
		print(f'*** 7 batch_graph_tensors.size(): {batch_graph_tensors.size()}')

		targets = [graph_label_list[idx] for idx in selected_idx]
		all_targets += targets

		# Get graph labels of all graphs in batch
		labels = torch.LongTensor(len(selected_idx))

		for i in range(len(selected_idx)):
			labels[i] = targets[i]

		if cmd_args.cuda == '1':
			batch_graph_tensors = batch_graph_tensors.cuda()
			labels = labels.cuda()

		# Perform training
		start_forward = time.perf_counter()

		# print('*** 7 node_feat: ', node_feat)
		# print('*** 8 n2n: ', n2n)

		# sys.exit()

		# dfscode_rnn_output = classifier.dfs_code_rnn(batch_graph_tensors)
		# output = classifier.output_layer(dfscode_rnn_output)
		output = classifier(batch_graph_tensors)
		output = torch.squeeze(output).unsqueeze(0)
		#print('** main.py line 88: output.is_cuda: ', output.is_cuda)
		temp_timing_dict["forward"].append(time.perf_counter() - start_forward)

		# TODO 5 
		# softmax, logits, prob, output

		# logits = F.log_softmax(output, dim=1)
		# prob = F.softmax(logits, dim=1)
		logits = prob = output
		print(f'output: {output}')
		print(f'logits: {logits}')
		print(f'prob: {prob}')
		print(f'labels: {labels}')
		logits = logits.float()
		labels = labels.float()

		loss = F.binary_cross_entropy(logits, labels)
		loss.backward()

		print('loss.backward()')
		sys.exit()

		loss = F.binary_cross_entropy(logits.float(), labels.float())
		pred = torch.tensor(1) if logits.squeeze()>0 else torch.tensor(0)
		# loss = F.nll_loss(logits, labels)
		# pred = logits.data.max(1, keepdim=True)[1]
		acc = pred.eq(labels.data.view_as(pred)).cpu().sum().item() /\
			  float(labels.size()[0])
		all_scores.append(prob.cpu().detach())  # for classification

		# Back propagate loss
		if optimizer is not None:
			with torch.autograd.set_detect_anomaly(True):
				optimizer.zero_grad()
				start_backward = time.perf_counter()
				loss.backward(retain_graph=True)
				temp_timing_dict["backward"].append(
					time.perf_counter() - start_backward)
				optimizer.step()

		loss = loss.data.cpu().detach().numpy()
		# print('loss: %0.5f acc: %0.5f' % (loss, acc))
		total_loss.append( np.array([loss, acc]) * len(selected_idx))

		n_samples += len(selected_idx)

	if optimizer is None:
		assert n_samples == len(g_list)

	# Calculate average loss and report performance metrics
	total_loss = np.array(total_loss)
	avg_loss = np.sum(total_loss, 0) / n_samples
	roc_auc, prc_auc = auc_scores(all_targets, all_scores)
	avg_loss = np.concatenate((avg_loss, [roc_auc], [prc_auc]))

	# Append loop average to global timer tracking list.
	# Only for training phase
	if optimizer is not None:
		timing_dict["forward"].append(
			sum(temp_timing_dict["forward"])/
			len(temp_timing_dict["forward"]))
		timing_dict["backward"].append(
			sum(temp_timing_dict["backward"])/
			len(temp_timing_dict["backward"]))
	
	return avg_loss

def loop_dataset(g_list, classifier, sample_idxes, config, dataset_features, optimizer=None):
	'''
	:param g_list: list of graphs to trainover
	:param classifier: the initialised classifier
	:param sample_idxes: indexes to mark the training and test graphs
	:param config: Run configurations as stated in config.yml
	:param dataset_features: Dataset features obtained from load_data.py
	:param optimizer: optimizer to use
	:return: average loss and other model performance metrics
	'''

	print('*** 4 len(g_list): ', len(g_list))
	print('*** 5 sample_idxes: ', sample_idxes)
	# print('*** 6 config: ', config)

	n_samples = 0
	all_targets = []
	all_scores = []
	total_loss = []

	# Determine batch size and initialise progress bar (pbar)
	bsize = max(config["general"]["batch_size"], 1)
	total_iters = (len(sample_idxes) + (bsize - 1) *
				   (optimizer is None)) // bsize
	# pbar = tqdm(range(total_iters), unit='batch')
	print(f'*** 6 total_iters: {total_iters}')

	# Create temporary timer dict to store timing data for this loop
	temp_timing_dict = {"forward": [], "backward": []}

	# For each batch
	for pos in range(total_iters):
		selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

		batch_graph = [g_list[idx] for idx in selected_idx]
		targets = [g_list[idx].label for idx in selected_idx]
		all_targets += targets

		node_feat, n2n, subg = graph_to_tensor(
			batch_graph, dataset_features["feat_dim"],
			dataset_features["edge_feat_dim"], cmd_args.cuda)

		# Get graph labels of all graphs in batch
		labels = torch.LongTensor(len(batch_graph))

		for i in range(len(batch_graph)):
			labels[i] = batch_graph[i].label

		if cmd_args.cuda == '1':
			#print('** main.py line 82: label cuda')
			labels = labels.cuda()

		# Perform training
		start_forward = time.perf_counter()

		# print('*** 7 node_feat: ', node_feat)
		# print('*** 8 n2n: ', n2n)

		# sys.exit()

		output = classifier(node_feat, n2n, subg, batch_graph)
		#print('** main.py line 88: output.is_cuda: ', output.is_cuda)
		temp_timing_dict["forward"].append(time.perf_counter() - start_forward)
		logits = F.log_softmax(output, dim=1)
		prob = F.softmax(logits, dim=1)

		# Calculate accuracy and loss
		#print('** main.py line 93: logits.is_cuda: ', logits.is_cuda)
		#print('** main.py line 94: labels.is_cuda: ', labels.is_cuda)
		loss = classifier.loss(logits, labels)
		pred = logits.data.max(1, keepdim=True)[1]
		acc = pred.eq(labels.data.view_as(pred)).cpu().sum().item() /\
			  float(labels.size()[0])
		all_scores.append(prob.cpu().detach())  # for classification

		# Back propagate loss
		if optimizer is not None:
			optimizer.zero_grad()
			start_backward = time.perf_counter()
			loss.backward()
			temp_timing_dict["backward"].append(
				time.perf_counter() - start_backward)
			optimizer.step()

		loss = loss.data.cpu().detach().numpy()
		# print('loss: %0.5f acc: %0.5f' % (loss, acc))
		total_loss.append( np.array([loss, acc]) * len(selected_idx))

		n_samples += len(selected_idx)

	if optimizer is None:
		assert n_samples == len(sample_idxes)

	# Calculate average loss and report performance metrics
	total_loss = np.array(total_loss)
	avg_loss = np.sum(total_loss, 0) / n_samples
	roc_auc, prc_auc = auc_scores(all_targets, all_scores)
	avg_loss = np.concatenate((avg_loss, [roc_auc], [prc_auc]))

	# Append loop average to global timer tracking list.
	# Only for training phase
	if optimizer is not None:
		timing_dict["forward"].append(
			sum(temp_timing_dict["forward"])/
			len(temp_timing_dict["forward"]))
		timing_dict["backward"].append(
			sum(temp_timing_dict["backward"])/
			len(temp_timing_dict["backward"]))
	
	return avg_loss

'''
	Main program execution
'''
if __name__ == '__main__':
	# Get run arguments
	cmd_opt = argparse.ArgumentParser(
		description='Argparser for graph classification')
	cmd_opt.add_argument('-cuda', default='1', help='0-CPU, 1-GPU')
	cmd_opt.add_argument('-gm', default='DFScodeRNN_cls', help='GNN model to use')
	cmd_opt.add_argument('-data', default='MUTAG', help='Dataset to use')
	# 0 -> Load classifier, 1 -> train from scratch
	cmd_opt.add_argument('-retrain', default='1', help='Whether to re-train the classifier or use saved trained model')
	cmd_args, _ = cmd_opt.parse_known_args()

	# Get run configurations
	config = yaml.safe_load(open("config.yml"))
	config["run"]["model"] = cmd_args.gm
	config["run"]["dataset"] = cmd_args.data

	# Set random seed
	random.seed(config["run"]["seed"])
	np.random.seed(config["run"]["seed"])
	torch.manual_seed(config["run"]["seed"])

	# graphgen args
	args=None

	# [1] Load graph data using util.load_data(), see util.py =========================================================
	# Specify the dataset to use and the number of folds for partitioning

	if config["run"]["model"]=='DFScodeRNN_cls':

		sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "models/graphgen")))
		sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "models/graphgen/bin")))


		# import graphgen functions, replace args
		from models.graphgen.main import *


		graph_list = get_graph_list()
		graph_label_list = get_graph_label_list()

		
		# TODO 2
		# 统一 dataset features 和 feature map, 因为后面会用到
		dataset_features = get_feature_map(graph_label_list)
		# dataset_features are the same as feature_map in graphgen main.py

		# here graphs are simply indices
		train_graphs, test_graphs = split_train_test(config["run"]["k_fold"], graph_list, graph_label_list)
		print(f'\n\ngraphgen args.__dict__: {pprint.pprint(args.__dict__)}\n\n')
		print(f'\n\ndataset_features: {dataset_features}\n\n')

	else:
		train_graphs, test_graphs, dataset_features = load_model_data(
			config["run"]["dataset"],
			config["run"]["k_fold"],
			config["general"]["data_autobalance"],
			config["general"]["print_dataset_features"]
		)

	config["dataset_features"] = dataset_features

	print(f'\n\nconfig: {config}\n\n')

	# [2] Instantiate the classifier using config.yml =================================================================
	# Display to user the current configuration used:
	run_configuration_string = "==== Configuration Settings ====\n"
	run_configuration_string += "== Run Settings ==\n"
	run_configuration_string += "Model: %s, Dataset: %s\n" % (
		config["run"]["model"], config["run"]["dataset"])

	for option, value in config["run"].items():
		run_configuration_string += "%s: %s\n" % (option, value)

	run_configuration_string += "\n== Model Settings and results ==\n"
	for option, value in config["GNN_models"][config["run"]["model"]].items():
		run_configuration_string += "%s: %s\n" % (option, value)
	run_configuration_string += "\n"

	run_statistics_string += run_configuration_string

	model_list = []
	model_metrics_dict = {"accuracy": [], "roc_auc": [], "prc_auc": []}

	# If execution is set to use existing model:
	# Hash the configurations
	# run_hash = hashlib.md5(
	# 	(json.dumps(config["run"], sort_keys=True).encode('utf-8'))).hexdigest()
	# model_hash = hashlib.md5(
	# 	json.dumps(config["GNN_models"][config["run"]["model"]], sort_keys=True).encode('utf-8')).hexdigest()

	if cmd_args.retrain == '0':
		# # Load classifier if it exists:
		# model_list = None
		# try:
		# 	model_list = torch.load(
		# 		"tmp/saved_models/%s_%s_%s_%s.pth" %
		# 		(dataset_features["name"], config["run"]["model"], run_hash, model_hash))

		# except FileNotFoundError:
		# 	print("Retrain is disabled but no such save of %s for dataset %s with the current configurations exists "
		# 		  "in tmp/saved_models folder. Please retry run with -retrain enabled." %
		# 		  (dataset_features["name"], config["run"]["model"]))
		# 	exit()

		# print("Testing models using saved model: " + config["run"]["model"])

		# # For each model trained on each fold
		# for fold_number in range(len(model_list)):
		# 	print("Testing using fold %s" % fold_number)
		# 	model_list[fold_number].eval()

		# 	# Get the test graph fold used in training the model
		# 	test_graph_fold = test_graphs[fold_number]
		# 	test_idxes = list(range(len(test_graph_fold)))

		# 	# Calculate test loss
		# 	test_loss = loop_dataset(test_graph_fold, model_list[fold_number],
		# 							 test_idxes, config, dataset_features)

		# 	# Print testing results for epoch
		# 	print('\033[93m'
		# 		  'average test: loss %.5f '
		# 		  'acc %.5f '
		# 		  'roc_auc %.5f '
		# 		  'prc_auc %.5f'
		# 		  '\033[0m' % (
		# 		test_loss[0], test_loss[1], test_loss[2], test_loss[3]))

		# 	# Append epoch statistics for reporting purposes
		# 	model_metrics_dict["accuracy"].append(test_loss[1])
		# 	model_metrics_dict["roc_auc"].append(test_loss[2])
		# 	model_metrics_dict["prc_auc"].append(test_loss[3])
		pass

	# Retrain a new set of models if no existing model exists or if retraining is forced
	else:
		print("Training a new model: " + config["run"]["model"])

		# [3] Begin training and testing ======================================
		fold_number = 0
		for train_graph_fold, test_graph_fold in \
				zip(train_graphs, test_graphs):
			print("Training model with dataset, testing using fold %s"
				  % fold_number)

			# load model
			if config["run"]["model"]=='DFScodeRNN_cls':
				classifier_model = get_model(dataset_features)
			else:
				exec_string = "classifier_model = %s(deepcopy(config[\"GNN_models\"][\"%s\"])," \
							" deepcopy(config[\"dataset_features\"]))" % \
							(config["run"]["model"], config["run"]["model"])
				exec (exec_string)

			if cmd_args.cuda == '1':
				classifier_model = classifier_model.cuda()

			# Define back propagation optimizer
			optimizer = optim.Adam(classifier_model.parameters(),
								   lr=config["run"]["learning_rate"])

			train_idxes = list(range(len(train_graph_fold)))
			test_idxes = list(range(len(test_graph_fold)))
			best_loss = None

			# For each epoch:
			for epoch in range(config["run"]["num_epochs"]):
				# Set classifier to train mode
				classifier_model.train()

				# Calculate training loss
				if config["run"]["model"]=='DFScodeRNN_cls':
					avg_loss = loop_dataset_DFSRNN(
					train_graph_fold, graph_label_list, classifier_model,
					config, args, dataset_features,
					optimizer=optimizer)
				else:
					avg_loss = loop_dataset(
						train_graph_fold, classifier_model,
						train_idxes, config, dataset_features,
						optimizer=optimizer)

				# Print training results for epoch
				print('\033[92m'
					  'average training of epoch %d: '
					  'loss %.5f '
					  'acc %.5f '
					  'roc_auc %.5f '
					  'prc_auc %.5f'
					  '\033[0m' % \
					(epoch, avg_loss[0], avg_loss[1],
					avg_loss[2], avg_loss[3]))

				# Set classifier to evaluation mode
				classifier_model.eval()

				# Calculate test loss

				if config["run"]["model"]=='DFScodeRNN_cls':
					test_loss = loop_dataset_DFSRNN(
					test_graph_fold, graph_label_list, classifier_model,
					config, args, dataset_features)
				else:
					test_loss = loop_dataset(
					test_graph_fold, classifier_model,
					test_idxes, config, dataset_features)

				# Print testing results for epoch
				print('\033[93m'
					  'average test of epoch %d: '
					  'loss %.5f '
					  'acc %.5f '
					  'roc_auc %.5f '
					  'prc_auc %.5f'
					  '\033[0m' % \
					  (epoch, test_loss[0], test_loss[1],
					  test_loss[2], test_loss[3]))

			# Append epoch statistics for reporting purposes
			model_metrics_dict["accuracy"].append(test_loss[1])
			model_metrics_dict["roc_auc"].append(test_loss[2])
			model_metrics_dict["prc_auc"].append(test_loss[3])

			# Append model to model list
			model_list.append(classifier_model)
			fold_number += 1

		# Save all models
		# print("Saving trained model %s for dataset %s" %
		# 	  (dataset_features["name"], config["run"]["model"]))
		# torch.save(model_list, "tmp/saved_models/%s_%s_%s_%s.pth" % \
		# 		   (dataset_features["name"],config["run"]["model"],run_hash,model_hash))

	# Report average performance metrics
	run_statistics_string += "Accuracy (avg): %s " % \
							 round(sum(model_metrics_dict["accuracy"])/len(model_metrics_dict["accuracy"]),5)
	run_statistics_string += "ROC_AUC (avg): %s " % \
							 round(sum(model_metrics_dict["roc_auc"])/len(model_metrics_dict["roc_auc"]),5)
	run_statistics_string += "PRC_AUC (avg): %s " % \
							 round(sum(model_metrics_dict["prc_auc"])/len(model_metrics_dict["prc_auc"]),5)
	run_statistics_string += "\n\n"
	'''
	# [4] Begin applying interpretability methods =====================================================================
	# Store the model that has the best ROC_AUC accuracy to
	# be used for generating saliency visualisations
	index_max_roc_auc = np.argmax(model_metrics_dict["roc_auc"])
	best_saliency_outputs_dict = {}

	saliency_map_generation_time_dict = {
		method: [] for method in config["interpretability_methods"].keys()}
	qualitative_metrics_dict_by_method = {
		method: {"fidelity": [], "contrastivity": [], "sparsity": []}
		for method in config["interpretability_methods"].keys()}

	print("Applying interpretability methods")

	# For each model trained on each fold
	for fold_number in range(len(model_list)):
		# For each enabled interpretability method
		for method in config["interpretability_methods"].keys():
			if config["interpretability_methods"][method]["enabled"] is True:
				print("Running method: %s for fold %s" %
					  (str(method), str(fold_number)))

				# Set up and run execution string
				exec_string = "score_output, saliency_output," \
							  " generate_score_execution_time = " \
							  "%s(model_list[fold_number], config," \
							  " dataset_features," \
							  " test_graphs[fold_number]," \
							  " fold_number," \
							  " cmd_args.cuda)" % method
				exec(exec_string)

				# If interpretability method is applied to the model with the
				# best roc_auc, save the attribution score
				if fold_number == index_max_roc_auc:
					best_saliency_outputs_dict.update(saliency_output)
				saliency_map_generation_time_dict[method].append(generate_score_execution_time)

				# Calculate qualitative metrics
				fidelity, contrastivity, sparsity = compute_metric(
					model_list[fold_number], score_output, dataset_features,config, cmd_args.cuda)

				qualitative_metrics_dict_by_method[method]["fidelity"].append(fidelity)
				qualitative_metrics_dict_by_method[method]["contrastivity"].append(contrastivity)
				qualitative_metrics_dict_by_method[method]["sparsity"].append(sparsity)

	# Report qualitative metrics and configuration used
	run_statistics_string += ("== Interpretability methods settings and results ==\n")
	for method, qualitative_metrics_dict in \
			qualitative_metrics_dict_by_method.items():
		if config["interpretability_methods"][method]["enabled"] is True:
			# Report configuration settings used
			run_statistics_string += \
				"Qualitative metrics and settings for method %s:\n " % \
				method
			for option, value in config["interpretability_methods"][method].items():
				run_statistics_string += "%s: %s\n" % (str(option), str(value))

			# Report qualitative metrics
			run_statistics_string += \
				"Fidelity (avg): %s " % \
				str(round(sum(qualitative_metrics_dict["fidelity"])/len(qualitative_metrics_dict["fidelity"]), 5))
			run_statistics_string += \
				"Contrastivity (avg): %s " % \
				str(round(
					sum(qualitative_metrics_dict["contrastivity"])/len(qualitative_metrics_dict["contrastivity"]), 5))
			run_statistics_string += \
				"Sparsity (avg): %s\n" % \
				str(round(sum(qualitative_metrics_dict["sparsity"])/len(qualitative_metrics_dict["sparsity"]), 5))
			run_statistics_string += \
				"Time taken to generate saliency scores: %s\n" % \
				str(round(sum(saliency_map_generation_time_dict[method])/
					len(saliency_map_generation_time_dict[method])*1000, 5))

			run_statistics_string += "\n"

	run_statistics_string += "\n\n"

	# [5] Create heatmap from the model with the best ROC_AUC output ==================================================
	custom_model_visualisation_options = None
	custom_dataset_visualisation_options = None

	# Sanity check:
	if config["run"]["model"] in \
			config["custom_visualisation_options"]["GNN_models"].keys():
		custom_model_visualisation_options = \
			config["custom_visualisation_options"]["GNN_models"][config["run"]["model"]]

	if config["run"]["dataset"] in \
			config["custom_visualisation_options"]["dataset"].keys():
		custom_dataset_visualisation_options = \
			config["custom_visualisation_options"]["dataset"][config["run"]["dataset"]]

	# Generate saliency visualistion images
	output_count = output_to_images(best_saliency_outputs_dict,
									dataset_features,
									custom_model_visualisation_options,
									custom_dataset_visualisation_options,
									output_directory="results/image")
	print("Generated %s saliency map images." % output_count)
	'''
	# [6] Print and log run statistics ========================================
	if len(timing_dict["forward"]) > 0:
		run_statistics_string += \
			"Average forward propagation time taken(ms): %s\n" % \
			str(sum(timing_dict["forward"])/len(timing_dict["forward"]) * 1000)
	if len(timing_dict["backward"]) > 0:
		run_statistics_string += \
			"Average backward propagation time taken(ms): %s\n" % \
			str(sum(timing_dict["backward"])/len(timing_dict["backward"]) * 1000)

	print(run_statistics_string)

	# Save dataset features and run statistics to log
	current_datetime = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
	log_file_name = "%s_%s_datetime_%s.txt" %\
				   (config["run"]["dataset"],
					config["run"]["model"],
					str(current_datetime))

	# Save log to text file
	with open("results/logs/%s" % log_file_name, "w") as f:
		# if "dataset_info" in dataset_features.keys():
		# 	dataset_info = dataset_features["dataset_info"] + "\n"
		# else:
		dataset_info = ""
		f.write(dataset_info + run_statistics_string)


