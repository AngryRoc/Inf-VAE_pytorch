import torch.nn.functional as F
from models.inits import *


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0, activation=F.relu):
        super(GCN, self).__init__()
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation
        self.dropout_rate = dropout_rate

    def forward(self, adj, features):
        hidden = F.dropout(features, self.dropout_rate, training=self.training)
        hidden = torch.mm(hidden, self.weight)
        hidden = torch.mm(adj, hidden)
        output = self.activation(hidden)
        return output
