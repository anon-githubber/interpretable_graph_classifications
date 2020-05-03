import numpy
import torch

from sklearn import metrics

def auc_scores(all_targets, all_scores):
	all_scores = torch.cat(all_scores).cpu().numpy()
	number_of_classes = int(all_scores.shape[1])

	# For binary classification:
	if number_of_classes == 2:
		# Take only second column (i.e. scores for positive label)
		all_scores = all_scores[:, 1]
		roc_auc = metrics.roc_auc_score(all_targets, all_scores, average='macro')
		prc_auc = metrics.average_precision_score(all_targets, all_scores, average='macro', pos_label=1)
	# For multi-class classification:
	if number_of_classes > 2:
		# Hand & Till (2001) implementation (ovo)
		roc_auc = metrics.roc_auc_score(all_targets, all_scores, multi_class='ovo', average='macro')


		exit()


	return roc_auc, prc_auc