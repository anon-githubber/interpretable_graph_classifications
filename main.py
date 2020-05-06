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

import argparse
from models import *
from interpretability_methods import *

from copy import deepcopy
from utilities.load_data import load_model_data
from utilities.util import graph_to_tensor
from utilities.output_results import output_to_images
from utilities.metrics import auc_scores, compute_metric

# Define timer list to report running statistics
timing_dict = {"forward": [], "backward": [], "generate_score": []}
run_statistics_string = ""

def loop_dataset(g_list, classifier, sample_idxes, config, dataset_features, optimizer=None):
	bsize = max(config["general"]["batch_size"], 1)

	total_loss = []
	total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
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
		temp_timing_dict["forward"].append(time.perf_counter() - start_forward)
		logits = F.log_softmax(output, dim=1)
		prob = F.softmax(logits, dim=1)

		# Calculate accuracy and loss
		loss = F.nll_loss(logits, labels)
		pred = logits.data.max(1, keepdim=True)[1]
		acc = pred.eq(labels.data.view_as(pred)).cpu().sum().item() / float(labels.size()[0])
		all_scores.append(prob.cpu().detach())  # for classification

		# Back propagation
		if optimizer is not None:
			optimizer.zero_grad()
			start_backward = time.perf_counter()
			loss.backward()
			temp_timing_dict["backward"].append(time.perf_counter() - start_backward)
			optimizer.step()


		loss = loss.data.cpu().detach().numpy()
		pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )
		total_loss.append( np.array([loss, acc]) * len(selected_idx))

		n_samples += len(selected_idx)
	if optimizer is None:
		assert n_samples == len(sample_idxes)
	total_loss = np.array(total_loss)
	avg_loss = np.sum(total_loss, 0) / n_samples

	roc_auc, prc_auc = auc_scores(all_targets, all_scores)
	avg_loss = np.concatenate((avg_loss, [roc_auc], [prc_auc]))

	# Append loop average to global timer tracking list. Only for training phase
	if optimizer is not None:
		timing_dict["forward"].append(sum(temp_timing_dict["forward"])/ len(temp_timing_dict["forward"]))
		timing_dict["backward"].append(sum(temp_timing_dict["backward"])/ len(temp_timing_dict["backward"]))
	
	return avg_loss

'''
	Main program execution
'''
if __name__ == '__main__':
	# Get run arguments
	cmd_opt = argparse.ArgumentParser(description='Argparser for graph classification')
	cmd_opt.add_argument('-cuda', default='0', help='0-CPU, 1-GPU')
	cmd_opt.add_argument('-gm', default='DGCNN', help='GNN model to use')
	cmd_opt.add_argument('-data', default='TOX21', help='Dataset to use')
	cmd_opt.add_argument('-retrain', default='0', help='Whether to re-train the classifier or use saved trained model')
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

	config["dataset_features"] = dataset_features

	# Instantiate the classifier using the configurations ==============================================================
	# Use saved model
	model_list = []
	model_metrics_dict = {"accuracy": [], "roc_auc": [], "prc_auc": []}

	if cmd_args.retrain == '0':
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

			model_metrics_dict["accuracy"].append(test_loss[1])
			model_metrics_dict["roc_auc"].append(test_loss[2])
			model_metrics_dict["prc_auc"].append(test_loss[3])

	# Retrain a new set of models
	else:
		print("Training a new model: " + cmd_args.gm)

		# Begin training ===============================================================================================
		fold_number = 0
		for train_graph_fold, test_graph_fold in zip(train_graphs, test_graphs):
			print("Training model with dataset, testing using fold %s" % fold_number)
			exec_string = "classifier_model = %s(deepcopy(config[\"GNN_models\"][\"%s\"])," \
						  " deepcopy(config[\"dataset_features\"]))" % (cmd_args.gm, cmd_args.gm)
			exec (exec_string)

			if cmd_args.cuda == '1':
				classifier_model = classifier_model.cuda()

			# Define back propagation optimizer
			optimizer = optim.Adam(classifier_model.parameters(), lr=config["run"]["learning_rate"])

			train_idxes = list(range(len(train_graph_fold)))
			test_idxes = list(range(len(test_graph_fold)))
			best_loss = None

			# For each epoch:
			for epoch in range(config["run"]["num_epochs"]):
				classifier_model.train()
				avg_loss = loop_dataset(train_graph_fold, classifier_model, train_idxes,
										config, dataset_features, optimizer=optimizer)
				print('\033[92maverage training of epoch %d: loss %.5f acc %.5f roc_auc %.5f prc_auc %.5f\033[0m' % (
					epoch, avg_loss[0], avg_loss[1], avg_loss[2],avg_loss[3]))

				classifier_model.eval()
				test_loss = loop_dataset(test_graph_fold, classifier_model, test_idxes,
										 config, dataset_features)
				print('\033[93maverage test of epoch %d: loss %.5f acc %.5f roc_auc %.5f prc_auc %.5f\033[0m' % (
					epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]))

			model_metrics_dict["accuracy"].append(test_loss[1])
			model_metrics_dict["roc_auc"].append(test_loss[2])
			model_metrics_dict["prc_auc"].append(test_loss[3])

			model_list.append(classifier_model)
			fold_number += 1

		# Save all models
		print("Saving trained model %s for dataset %s" % (dataset_features["name"], cmd_args.gm))
		torch.save(model_list, "tmp/saved_models/%s_%s_epochs_%s_learnrate_%s_folds_%s.pth" %
				   (dataset_features["name"], cmd_args.gm, str(config["run"]["num_epochs"]),
					str(config["run"]["learning_rate"]), str(config["run"]["k_fold"])))

	run_statistics_string += "Average Accuracy: %s " % \
							 round(sum(model_metrics_dict["accuracy"])/len(model_metrics_dict["accuracy"]),5)
	run_statistics_string += "Average ROC_AUC: %s " % \
							 round(sum(model_metrics_dict["roc_auc"])/len(model_metrics_dict["roc_auc"]),5)
	run_statistics_string += "Average PRC_AUC: %s " % \
							 round(sum(model_metrics_dict["prc_auc"])/len(model_metrics_dict["prc_auc"]),5)

	# Begin applying interpretability methods ==========================================================================
	index_max_roc_auc = np.argmax(model_metrics_dict["roc_auc"])
	best_saliency_outputs_dict = {}

	print("Applying interpretability methods")
	for fold_number in range(len(model_list)):
		for method in config["interpretability_methods"].keys():
			if config["interpretability_methods"][method]["enabled"] is True:
				print("Running method: %s for fold %s" % (str(method), str(fold_number)))
				exec_string = "score_output, saliency_output, generate_score_execution_time = " \
							  "%s(model_list[fold_number], config[\"interpretability_methods\"][\"%s\"]," \
							  " dataset_features, test_graphs[fold_number], cmd_args.cuda)" % (method, method)
				exec(exec_string)

				if fold_number == index_max_roc_auc:
					best_saliency_outputs_dict.update(saliency_output)
		timing_dict["generate_score"].append(generate_score_execution_time)

		# Calculate qualitative metrics ================================================================================
		#metric_outputs = compute_metric(model_list[fold_number], test_graphs[fold_number], score_output, \
		 		dataset_features, config, cmd_args.cuda)

	# Create heatmap from the model with the best ROC_AUC output =======================================================
	output_count = output_to_images(saliency_output, dataset_features, output_directory="results/image")

	print("Generated %s saliency map images." % output_count)


	# Print run statistics =============================================================================================
	if len(timing_dict["forward"]) > 0:
		run_statistics_string += "Average forward propagation time taken(ms): %s\n" %\
								 str(sum(timing_dict["forward"])/len(timing_dict["forward"]) * 1000)
	if len(timing_dict["backward"]) > 0:
		run_statistics_string += "Average backward propagation time taken(ms): %s\n" %\
								 str(sum(timing_dict["backward"])/len(timing_dict["backward"])* 1000)
	run_statistics_string += "Average time taken to generate attribution scores(ms): %s\n" %\
							 str(sum(timing_dict["generate_score"])/len(timing_dict["generate_score"]) * 1000)

	print(run_statistics_string)

