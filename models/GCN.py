import torch
import torch.nn as nn
import math
from models.lib.weight_util import weights_init
from torch.nn.parameter import Parameter
from models.layers.graph_convolution_layers import GraphConvolutionLayers_GCN

class GCN(nn.Module):
	def __init__(self, config, dataset_features, regression=False):
		super(GCN, self).__init__()
		self.regression = regression
		self.config = config
		self.dataset_features = dataset_features

		# Initialise Graph Convolution Layers
		self.config["convolution_layers_size"] = \
			list(map(int, self.config["convolution_layers_size"].split('-')))

		self.graph_convolution = GraphConvolutionLayers_GCN(
			latent_dim=self.config["convolution_layers_size"],
			num_node_feats=dataset_features["feat_dim"] + dataset_features["attr_dim"],
			num_edge_feats=dataset_features["edge_feat_dim"], dropout=self.config["dropout"],
			concat_tensors=False)

		self.weight = Parameter(torch.FloatTensor(config["convolution_layers_size"][-1],
												  dataset_features["num_class"]))

		weights_init(self)

	def forward(self, node_feat, n2n, subg, batch_graph):
		graph_sizes = [batch_graph[i].number_of_nodes for i in range(len(batch_graph))]
		output_matrix = self.graph_convolution(node_feat, n2n, batch_graph)

		batch_logits = torch.zeros(len(graph_sizes), self.dataset_features["num_class"])

		accum_count=0
		for i in range(subg):
			to_pool = output_matrix[accum_count:accum_count+graph_sizes[i]]
			average_pooling = to_pool.mean(0, keepdim=True)
			pool_out = average_pooling.mm(self.weight)
			batch_logits[i] = pool_out

		return batch_logits

	def output_features(self, batch_graph):
		embed = self.graph_convolution(batch_graph)
		return embed, labels