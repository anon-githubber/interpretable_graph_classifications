import torch
import torch.nn as nn
from models.lib.weight_util import weights_init
from models.layers.graph_convolution_layers import GraphConvolutionLayers_GraphSAGE
from models.layers.softassign_graph_convolution_layers import SoftAssignGraphConvolutionLayer
from models.layers.mlp_layers import MLPClassifier

class DiffPool(nn.Module):
	def __init__(self, config, dataset_features, regression=False):
		super(DiffPool, self).__init__()
		self.regression = regression
		self.config = config
		self.dataset_features = dataset_features

		# Initialise Graph Convolution Layers
		self.config["convolution_layers_size"] = \
			list(map(int, self.config["convolution_layers_size"].split('-')))

		self.config["assign_dim"] = int(dataset_features["max_num_nodes"] * config["assign_ratio"])

		self.graph_convolution = GraphConvolutionLayers_GraphSAGE(
			latent_dim=self.config["convolution_layers_size"],
			num_node_feats=dataset_features["feat_dim"]+dataset_features["attr_dim"],
			num_edge_feats=dataset_features["edge_feat_dim"], concat_tensors=True)

		self.graph_assign = GraphConvolutionLayers_GraphSAGE(
			latent_dim=self.config["convolution_layers_size"],
			num_node_feats=dataset_features["feat_dim"] + dataset_features["attr_dim"],
			num_edge_feats=dataset_features["edge_feat_dim"], concat_tensors=True)

		# self.softassign_graph_convolution = SoftAssignGraphConvolutionLayer(
		# 	latent_dim=self.config["convolution_layers_size"],
		# 	num_node_feats=dataset_features["feat_dim"]+dataset_features["attr_dim"],
		# 	num_edge_feats=dataset_features["edge_feat_dim"],
		# 	assign_dim=self.config["assign_dim"])

	def forward(self, node_feat, adjacency_matrix, subg_size, batch_graph):
		graph_sizes = [batch_graph[i].number_of_nodes for i in range(len(batch_graph))]

		embedding_tensor = self.graph_convolution(node_feat, adjacency_matrix, batch_graph)

		print(embedding_tensor.size())
		out, _ = torch.max(embedding_tensor, dim=0)
		print(out.size())
		exit()
		#assign_tensor = self.softassign_graph_convolution(node_feat, adjacency_matrix, batch_graph)

		print(embedding_tensor.size())
		print(assign_tensor.size())
		exit()
		return

	def output_features(self, batch_graph):
		embed = self.graph_convolution(batch_graph)
		return embed, labels