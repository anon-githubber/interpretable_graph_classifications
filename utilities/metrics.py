import numpy
import torch
from utilities.fidelity import measure_fidelity
from utilities.contrastivity import measure_contrastivity
from utilities.sparsity import measure_sparsity
from copy import deepcopy
from sklearn import metrics


def auc_scores(all_targets, all_scores):
	all_scores = torch.cat(all_scores).cpu().numpy()
	number_of_classes = int(all_scores.shape[1])

	# For binary classification:
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

		exit()

	return roc_auc, prc_auc


def compute_metric(metric, sampled_graphs, classifier_model, method, dataset_features, output, config, importance_ranges, use_gpu):
	if metric == 'fidelity':
		return measure_fidelity(classifier_model, deepcopy(sampled_graphs), output, dataset_features, use_gpu=use_gpu)
	elif metric == 'contrastivity':
		return measure_contrastivity(classifier_model, sampled_graphs, method, dataset_features, config)
	elif metric == 'sparsity':
		return measure_sparsity(classifier_model, sampled_graphs, method, dataset_features, config)
