import torch
import json
import random

from time import perf_counter
from os import path
from copy import deepcopy
from captum.attr import Saliency
from utilities.util import graph_to_tensor, normalize_scores

def GradCAM(classifier_model, config, dataset_features, GNNgraph_list, current_fold=None, cuda=0):
	'''
		:param classifier_model: trained classifier model
		:param config: parsed configuration file of config.yml
		:param dataset_features: a dictionary of dataset features obtained from load_data.py
		:param GNNgraph_list: a list of GNNgraphs obtained from the dataset
		:param current_fold: has no use in this method
		:param cuda: whether to use GPU to perform conversion to tensor
	'''
	# Initialise settings
	config = config
	interpretability_config = config["interpretability_methods"]["gradcam"]
	dataset_features = dataset_features

	# Perform deeplift on the classifier model
	gc = Saliency(classifier_model, classifier_model.graph_convolution)

	output_for_metrics_calculation = []
	output_for_generating_saliency_map = {}

	# Obtain attribution score for use in qualitative metrics
	tmp_timing_list = []

	for GNNgraph in GNNgraph_list:
		output = {'graph': GNNgraph}
		for _, label in dataset_features["label_dict"].items():
			# Relabel all just in case, may only relabel those that need relabelling
			# if performance is poor
			original_label = GNNgraph.label
			GNNgraph.label = label

			node_feat, n2n, subg = graph_to_tensor(
				[GNNgraph], dataset_features["feat_dim"],
				dataset_features["edge_feat_dim"], cuda)

			subg = subg.size()[0]
			start_generation = perf_counter()
			attribution = gc.attribute(node_feat,
								   additional_forward_args=(n2n, subg, [GNNgraph]),
								   target=label)
			attribution_score = torch.sum(attribution, dim=1)
			tmp_timing_list.append(perf_counter() - start_generation)
			attribution_score = normalize_scores(attribution_score, -1, 1)

			GNNgraph.label = original_label

			output[label] = attribution_score
		output_for_metrics_calculation.append(output)

	execution_time = sum(tmp_timing_list)/(len(tmp_timing_list))

	# Obtain attribution score for use in generating saliency map for comparison with zero tensors
	if interpretability_config["number_of_samples"] > 0:
		# Randomly sample from existing list:
		graph_idxes = list(range(len(output_for_metrics_calculation)))
		random.shuffle(graph_idxes)
		output_for_generating_saliency_map.update({"gradcam_%s" % str(label): []
												   for _, label in dataset_features["label_dict"].items()})

		# Begin appending found samples
		for index in graph_idxes:
			tmp_label = output_for_metrics_calculation[index]['graph'].label
			if len(output_for_generating_saliency_map["gradcam_%s" % str(tmp_label)]) < \
				interpretability_config["number_of_samples"]:
				output_for_generating_saliency_map["gradcam_%s" % str(tmp_label)].append(
					(output_for_metrics_calculation[index]['graph'], output_for_metrics_calculation[index][tmp_label]))

	return output_for_metrics_calculation, output_for_generating_saliency_map, execution_time