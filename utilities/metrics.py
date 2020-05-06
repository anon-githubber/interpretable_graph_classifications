import numpy
import torch

import torch.nn.functional as F
from copy import deepcopy
from sklearn import metrics

from utilities.util import graph_to_tensor, hamming

def auc_scores(all_targets, all_scores):
	all_scores = torch.cat(all_scores).cpu().numpy()
	number_of_classes = int(all_scores.shape[1])

	# For binary classification:
	roc_auc = 0.0
	prc_auc = 0.0
	if number_of_classes == 2:
		# Take only second column (i.e. scores for positive label)
		all_scores = all_scores[:, 1]
		roc_auc = metrics.roc_auc_score(
			all_targets, all_scores, average='macro')
		prc_auc = metrics.average_precision_score(
			all_targets, all_scores, average='macro', pos_label=1)
	# For multi-class classification:
	if number_of_classes > 2:
		# Hand & Till (2001) implementation (ovo)
		roc_auc = metrics.roc_auc_score(
			all_targets, all_scores, multi_class='ovo', average='macro')

		# TODO: build PRC-AUC calculations for multi-class datasets

	return roc_auc, prc_auc


def compute_metric(trained_classifier_model, GNNGraph_list, metric_attribution_score, dataset_features, config, cuda):
	if config["metrics"]["fidelity"]["enabled"] is True:
		fidelity_metric = get_fidelity(trained_classifier_model, metric_attribution_score, dataset_features,
						config, cuda)


def get_accuracy(trained_classifier_model, GNNgraph_list, dataset_features, cuda):
	trained_classifier_model.eval()
	true_equal_pred_pairs = []

	# Instead of sending the whole list as batch, do it one by one in case classifier do not support batch-processing
	# TODO: Enable batch processing support
	for GNNgraph in GNNgraph_list:
		node_feat, n2n, subg = graph_to_tensor(
            [GNNgraph], dataset_features["feat_dim"],
            dataset_features["edge_feat_dim"], cuda)

		subg = subg.size()[0]

		output = trained_classifier_model(node_feat, n2n, subg, [GNNgraph])
		logits = F.log_softmax(output, dim=1)
		pred = logits.data.max(1, keepdim=True)[1]

		if GNNgraph.label == int(pred[0]):
			true_equal_pred_pairs.append(1)
		else:
			true_equal_pred_pairs.append(0)

	return sum(true_equal_pred_pairs)/len(true_equal_pred_pairs)

def is_salient(score, importance_range):
	start, end = importance_range
	if start <= score <= end:
		return True
	else:
		return False

def occlude_graphs(metric_attribution_scores, config):
	# Transform the graphs, occlude nodes with significant attribution scores
	importance_range = config["metrics"]["fidelity"]["importance_range"].split(",")
	importance_range = [float(bound) for bound in importance_range]

	occluded_GNNgraph_list = []
	for group in metric_attribution_scores:
		GNNgraph = deepcopy(group['graph'])
		attribution_score = group[GNNgraph.label]

		for i in range(len(attribution_score)):
			if is_salient(abs(float(attribution_score[i])), importance_range):
				GNNgraph.node_labels[i] = None
		occluded_GNNgraph_list.append(GNNgraph)
	return occluded_GNNgraph_list


def get_fidelity(trained_classifier_model, metric_attribution_scores, dataset_features, config, cuda):
	GNNgraph_list = [group["graph"] for group in metric_attribution_scores]

	accuracy_prior_occlusion = get_accuracy(trained_classifier_model, GNNgraph_list, dataset_features, cuda)
	occluded_GNNgraph_list = occlude_graphs(metric_attribution_scores, config)
	accuracy_after_occlusion = get_accuracy(trained_classifier_model, occluded_GNNgraph_list, dataset_features, cuda)
	print(accuracy_after_occlusion)
	exit()
