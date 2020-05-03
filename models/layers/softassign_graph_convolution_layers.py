import torch
import torch.nn as nn
import torch.nn.functional as F
import models.layers.softassign_prediction_embedding

class SoftAssignGraphConvolutionLayer(nn.Module):
	'''
		Graph Convolution layers with Soft Assign for hierarchical clustering
	'''

	def __init__(self,
				 input_dim,
				 assign_dim,
				 assign_pred_hidden_dim,
				 label_dim,
				 add_self=False,
				 normalize_embedding=False,
				 dropout=0.0,
				 bias=True):

		super(SoftAssignGraphConvolutionLayers, self).__init__()
		self.add_self = add_self
		self.dropout = dropout
		self.normalize_embedding = normalize_embedding
		self.assign_dim = assign_dim
		self.assign_pred_hidden_dim = assign_pred_hidden_dim

		self.latent_dim = latent_dim
		self.assign_pred = SoftAssignPredictionEmbedding(input_dim, hidden_dim, label_dim)

	def forward(self, node_feat, n2n, batch_graph):
		pass
