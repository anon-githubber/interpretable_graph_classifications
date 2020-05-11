import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

from models.layers.graph_convolution_layer import GraphConvolutionLayer_GraphSAGE
from models.layers.graph_convolution_layers import GraphConvolutionLayers_GraphSAGE
from models.layers.dense_layers import DenseLayers

class DiffPool(nn.Module):
	def __init__(self, config, dataset_features, regression=False):
		super(DiffPool, self).__init__()
		self.regression = regression
		self.config = config
		self.dataset_features = dataset_features
		self.concat_tensors = config["concat_tensors"]
		self.num_pooling = config["number_of_pooling"]
		self.assign_ratio = config["assign_ratio"]
		self.linkpred = config["link_prediction"]
		self.input_dim = dataset_features["feat_dim"] + dataset_features["attr_dim"] + dataset_features["edge_feat_dim"]

		# Embedding Tensor
		self.config["convolution_layers_size"] = \
			list(map(int, self.config["convolution_layers_size"].split('-')))

		self.graph_convolution = GraphConvolutionLayers_GraphSAGE(
			latent_dim=self.config["convolution_layers_size"],
			input_dim=self.input_dim,
			concat_tensors=self.concat_tensors)

		if self.concat_tensors is True:
			self.pred_input_dim = sum(self.config["convolution_layers_size"])
		else:
			self.pred_input_dim = self.config["convolution_layers_size"][-1]

		# DiffPool Layers
		self.conv_modules = nn.ModuleList()
		self.assign_modules = nn.ModuleList()
		self.assign_pred_modules = nn.ModuleList()

		# Initialise first assign dimension
		assign_input_dim = self.input_dim
		assign_dim = int(self.dataset_features["max_num_nodes"] * self.config["assign_ratio"])

		for stack in range(self.num_pooling):
			# GNN Pool
			self.conv_modules.append(GraphConvolutionLayers_GraphSAGE(
				latent_dim=self.config["convolution_layers_size"],
				input_dim=self.pred_input_dim,
				concat_tensors=self.concat_tensors))

			# GNN Assign
			self.assign_modules.append(GraphConvolutionLayers_GraphSAGE(
				latent_dim=self.config["convolution_layers_size"] + [assign_dim],
				input_dim=assign_input_dim,
				concat_tensors=self.concat_tensors))

			if self.concat_tensors is True:
				assign_pred_input_dim = sum(self.config["convolution_layers_size"]) + assign_dim
			else:
				assign_pred_input_dim = assign_dim
			self.assign_pred_modules.append(DenseLayers(assign_pred_input_dim, assign_dim, []))

			# For next pooling stack
			assign_input_dim = self.pred_input_dim
			assign_dim = int(assign_dim * self.assign_ratio)

		# Prediction Layers
		self.config["pred_hidden_layers"] = \
			list(map(int, self.config["pred_hidden_layers"].split('-')))

		self.prediction_model = DenseLayers(self.pred_input_dim * (self.num_pooling+1), dataset_features["num_class"],
											self.config["pred_hidden_layers"])

		# Initialise weights
		for m in self.modules():
			if isinstance(m, GraphConvolutionLayer_GraphSAGE):
				m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
				if m.bias is not None:
					m.bias.data = init.constant_(m.bias.data, 0.0)

	def forward(self, node_feat, adjacency_matrix, subg_size, batch_graph):
		graph_sizes = [batch_graph[i].number_of_nodes for i in range(len(batch_graph))]
		adjacency_matrix = adjacency_matrix.to_dense()
		self.input_adj = adjacency_matrix

		node_feat_a = node_feat
		out_all = []

		# Embedding Tensor
		embedding_tensor = self.graph_convolution(node_feat, adjacency_matrix, batch_graph)
		out, _ = torch.max(embedding_tensor, dim=0, keepdim=True)
		out_all.append(out)

		for stack in range(self.num_pooling):

			assign_tensor = self.assign_modules[stack](node_feat_a, adjacency_matrix, batch_graph)

			self.assign_tensor = nn.Softmax(dim	=-1)(self.assign_pred_modules[stack](assign_tensor))

			# update pooled features and adj matrix
			node_feat = torch.matmul(torch.transpose(assign_tensor, 0, 1), embedding_tensor)
			adjacency_matrix = torch.transpose(assign_tensor, 0, 1) @ adjacency_matrix @ assign_tensor
			node_feat_a = node_feat

			embedding_tensor = self.conv_modules[stack](node_feat, adjacency_matrix, batch_graph)

			out, _ = torch.max(embedding_tensor, dim=0, keepdim=True)
			out_all.append(out)

		if self.concat_tensors:
			output = torch.cat(out_all, dim=1)
		else:
			output = out

		pred = self.prediction_model(output)

		return pred

	def loss(self, logits, labels, type='softmax', adj_hop=1):
		'''
		Args:
			batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
		'''
		eps = 1e-7
		loss = F.cross_entropy(logits, labels, reduction='mean')

		if self.linkpred:
			adj = self.input_adj
			max_num_nodes = self.dataset_features["max_num_nodes"]

			pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 0, 1)
			tmp = pred_adj0
			pred_adj = pred_adj0
			for _ in range(adj_hop-1):
				tmp = tmp @ pred_adj0
				pred_adj = pred_adj + tmp

			pred_adj = torch.min(pred_adj, torch.Tensor(1))

			self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
			num_entries = max_num_nodes * max_num_nodes * adj.size()[0]

			self.link_loss = torch.sum(self.link_loss) / float(num_entries)

			#print('linkloss: ', self.link_loss)
			return loss + self.link_loss
		return loss
