import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GCN
from utils.preprocess import embedding_lookup
import math


class InfVAECascades(nn.Module):
    def __init__(self, num_nodes, latent_dim, max_seq_length,device):
        super(InfVAECascades, self).__init__()
        sqrt3 = math.sqrt(3)
        self.device=device
        self.num_nodes = num_nodes
        self.embedding_size = latent_dim
        self.max_seq_length = max_seq_length
        self.co_attn_wts = nn.Parameter((sqrt3 + sqrt3) * torch.rand(self.embedding_size, self.embedding_size) - sqrt3)
        self.temporal_embeddings = nn.Parameter(
            (sqrt3 + sqrt3) * torch.rand(self.num_nodes, self.embedding_size) - sqrt3)

        self.receiver_embeddings = nn.Parameter(
            (sqrt3 + sqrt3) * torch.rand(self.num_nodes, self.embedding_size) - sqrt3)

        self.sender_embeddings = nn.Parameter((sqrt3 + sqrt3) * torch.rand(self.num_nodes, self.embedding_size) - sqrt3)

        self.position_embeddings = nn.Parameter(
            (sqrt3 + sqrt3) * torch.rand(self.max_seq_length, self.embedding_size) - sqrt3)

    def forward(self, examples, masks):
        sender_embedded = embedding_lookup(self.sender_embeddings, examples).to(self.device)

        temporal_embedded = embedding_lookup(self.temporal_embeddings, examples).to(self.device)

        # Mask input sequence.
        sender_embedded = torch.multiply(sender_embedded, torch.unsqueeze(masks.float(), -1))
        temporal_embedded = torch.multiply(temporal_embedded, torch.unsqueeze(masks.float(), -1))
        temporal_embedded = temporal_embedded + self.position_embeddings

        attn_act = torch.tanh(
            torch.sum(
                torch.multiply(
                    torch.tensordot(sender_embedded, self.co_attn_wts, dims=([2], [0])),
                    temporal_embedded
                ), 2
            )
        )
        attn_alpha = nn.Softmax(dim=1)(attn_act)
        attended_embeddings = torch.multiply(temporal_embedded, torch.unsqueeze(attn_alpha, -1))
        attended_embeddings = torch.sum(attended_embeddings, 1)  # (batch_size, embed_size)
        outputs = torch.matmul(attended_embeddings, self.receiver_embeddings.T)
        return outputs


class InfVAESocial(nn.Module):

    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, latent_dim, device, dropout_rate=0):
        super(InfVAESocial, self).__init__()
        self.device = device
        self.gcn1 = GCN(input_dim, hidden1_dim, dropout_rate)
        self.gcn2 = GCN(hidden1_dim, hidden2_dim, dropout_rate)
        self.gcn3 = GCN(hidden2_dim, hidden3_dim, dropout_rate)
        self.gcn_mean = GCN(hidden3_dim, latent_dim, dropout_rate)
        self.gcn_logstd = GCN(hidden3_dim, latent_dim, dropout_rate, activation=F.softplus)

    def encode(self, adj, features):
        hidden = self.gcn1(adj, features)
        hidden = self.gcn2(adj, hidden)
        hidden = self.gcn3(adj, hidden)
        self.mean = self.gcn_mean(adj, hidden)
        self.logstd = self.gcn_logstd(adj, hidden)
        gaussian_noise = torch.randn(self.mean.shape[0], self.mean.shape[1]).to(self.device)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, adj, features):
        Z = self.encode(adj, features)
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
