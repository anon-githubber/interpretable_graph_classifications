import torch
import torch.nn as nn
from models.layers.lib.layer_util import gnn_spmm

class SoftAssignPredictionEmbedding(nn.Module):
	def __init__(self,
				 input_dim,
				 hidden_dim,
				 label_dim):
		super(SoftAssignLayer, self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.label_dim = label_dim

		if len(pred_hidden_dims) == 0:
			self.pred_model = nn.Linear(pred_input_dim, label_dim)
		else:
			pred_layers = []
			for pred_dim in hidden_dim:
				pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
				pred_layers.append(self.act)
				pred_input_dim = pred_dim
			pred_layers.append(nn.Linear(pred_dim, label_dim))
			self.pred_model = nn.Sequential(*pred_layers)

	def forward(self, assign_tensor):
		return self.pred_model(assign_tensor)
