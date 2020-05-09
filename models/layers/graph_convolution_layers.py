import torch
import torch.nn as nn
from models.layers.graph_convolution_layer import GraphConvolutionLayer_GCN, GraphConvolutionLayer_DGCNN, GraphConvolutionLayer_GraphSAGE

# Torch nn module: Graph Convolution Layers using GCN implementation
class GraphConvolutionLayers_GCN(nn.Module):
	'''
		Graph Convolution layers
	'''
	def __init__(self,
		num_node_feats,
		num_edge_feats,
		latent_dim=[128, 256, 512],
		concat_tensors=False,
		dropout=0.0):

		print('Initializing Graph Convolution Layers')

		# Intialise settings
		super(GraphConvolutionLayers_GCN, self).__init__()
		self.latent_dim = latent_dim
		self.num_node_feats = num_node_feats
		self.num_edge_feats = num_edge_feats
		self.total_latent_dim = sum(latent_dim)
		self.concat_tensors = concat_tensors

		# Create convolution Layers as module list
		self.conv_layers = nn.ModuleList()

		# First layer takes in the node feature X
		self.conv_layers.append(GraphConvolutionLayer_GCN(num_node_feats + num_edge_feats,
													  latent_dim[0],
													  dropout))

		# Following layers take latent dim of previous convolution layer as input
		for i in range(1, len(latent_dim)):
			self.conv_layers.append(GraphConvolutionLayer_GCN(latent_dim[i-1], latent_dim[i], dropout))

	def forward(self, node_feat, adjacency_matrix, batch_graph):
		node_degs = [torch.Tensor(batch_graph[i].node_degrees) + 1 for i in range(len(batch_graph))]
		node_degs = torch.cat(node_degs).unsqueeze(1)

		# Graph Convolution Layers Forward
		lv = 0
		output_matrix = node_feat
		cat_output_matrix = []
		while lv < len(self.latent_dim):
			output_matrix = self.conv_layers[lv](output_matrix, adjacency_matrix, node_degs)
			cat_output_matrix.append(output_matrix)
			lv += 1

		if self.concat_tensors:
			return torch.cat(cat_output_matrix, 1)
		else:
			return output_matrix

# Torch nn module: Graph Convolution Layers using DGCNN implementation
class GraphConvolutionLayers_DGCNN(nn.Module):
	'''
		Graph Convolution layers
	'''
	def __init__(self,
		num_node_feats,
		num_edge_feats,
		latent_dim=[32, 32, 32, 1],
		concat_tensors=False,
		dropout=0.0):

		print('Initializing Graph Convolution Layers')

		# Intialise settings
		super(GraphConvolutionLayers_DGCNN, self).__init__()
		self.latent_dim = latent_dim
		self.num_node_feats = num_node_feats
		self.num_edge_feats = num_edge_feats
		self.total_latent_dim = sum(latent_dim)
		self.concat_tensors = concat_tensors

		# Create convolution Layers as module list
		self.conv_layers = nn.ModuleList()

		# First layer takes in the node feature X
		self.conv_layers.append(GraphConvolutionLayer_DGCNN(num_node_feats + num_edge_feats,
													  latent_dim[0],
													  dropout))

		# Following layers take latent dim of previous convolution layer as input
		for i in range(1, len(latent_dim)):
			self.conv_layers.append(GraphConvolutionLayer_DGCNN(latent_dim[i-1], latent_dim[i], dropout))

	def forward(self, node_feat, adjacency_matrix, batch_graph):
		node_degs = [torch.Tensor(batch_graph[i].node_degrees) + 1 for i in range(len(batch_graph))]
		node_degs = torch.cat(node_degs).unsqueeze(1)

		# Graph Convolution Layers Forward
		lv = 0
		output_matrix = node_feat
		cat_output_matrix= []
		while lv < len(self.latent_dim):
			output_matrix = self.conv_layers[lv](output_matrix, adjacency_matrix, node_degs)
			cat_output_matrix.append(output_matrix)
			lv += 1

		if self.concat_tensors:
			return torch.cat(cat_output_matrix, 1)
		else:
			return output_matrix

# Torch nn module: Graph Convolution Layers using GraphSAGE implementation
class GraphConvolutionLayers_GraphSAGE(nn.Module):
	'''
		Graph Convolution layers
	'''
	def __init__(self,
		num_node_feats,
		num_edge_feats,
		latent_dim=[32, 32, 32, 1],
		concat_tensors=False,
		dropout=0.0):

		print('Initializing Graph Convolution Layers')

		# Intialise settings
		super(GraphConvolutionLayers_GraphSAGE, self).__init__()
		self.latent_dim = latent_dim
		self.num_node_feats = num_node_feats
		self.num_edge_feats = num_edge_feats
		self.total_latent_dim = sum(latent_dim)
		self.concat_tensors = concat_tensors

		# Create convolution Layers as module list
		self.conv_layers = nn.ModuleList()

		# First layer takes in the node feature X
		self.conv_layers.append(GraphConvolutionLayer_GraphSAGE(num_node_feats + num_edge_feats,
													  latent_dim[0],
													  dropout))

		# Following layers take latent dim of previous convolution layer as input
		for i in range(1, len(latent_dim)):
			self.conv_layers.append(GraphConvolutionLayer_GraphSAGE(latent_dim[i-1], latent_dim[i], dropout))

	def forward(self, node_feat, adjacency_matrix, batch_graph):
		# Graph Convolution Layers Forward
		lv = 0
		output_matrix = node_feat
		cat_output_matrix = []
		while lv < len(self.latent_dim):
			output_matrix = self.conv_layers[lv](output_matrix, adjacency_matrix)
			cat_output_matrix.append(output_matrix)
			lv += 1

		if self.concat_tensors:
			return torch.cat(cat_output_matrix, 1)
		else:
			return output_matrix