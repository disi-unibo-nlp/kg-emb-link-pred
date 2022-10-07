from tokenize import Funny
from typing import Optional

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter, ReLU
from torch_scatter import scatter_add
from torch_sparse import SparseTensor

from torch_geometric.nn import RGCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, ones, zeros
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.utils import softmax


class GNNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, input_channels, hidden_channels, output_channels, num_relations, gnn_model):
        super().__init__()
        self.node_emb = Parameter(torch.Tensor(num_nodes, input_channels))
        self.gnn_model = gnn_model

        if gnn_model == 'rgat':
            self.conv1 = GATConv(input_channels, hidden_channels, edge_dim=num_relations)
            self.conv2 = GATConv(hidden_channels, output_channels, edge_dim=num_relations)
            emb_matrix = torch.eye(num_relations)
            self.one_hot_embedding = torch.nn.Embedding.from_pretrained(emb_matrix, freeze = True)

        elif gnn_model == 'rgcn':
            self.conv1 = RGCNConv(input_channels, hidden_channels, num_relations, num_blocks=5)
            self.conv2 = RGCNConv(hidden_channels, output_channels, num_relations, num_blocks=5)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, edge_type):
        
        x = self.node_emb

        if self.gnn_model == 'rgat':
            edge_attr = self.one_hot_embedding.weight[edge_type]
            x = torch.utils.checkpoint.checkpoint(self.conv1, x, edge_index, edge_attr)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv2(x, edge_index, edge_attr)

        elif self.gnn_model == 'rgcn':
            x = torch.utils.checkpoint.checkpoint(self.conv1, x, edge_index, edge_type)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv2(x, edge_index, edge_type)

        return x


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, input_channels):
        super().__init__()
        self.rel_emb = Parameter(torch.Tensor(num_relations, input_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)
    


class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases)

        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type, edge_norm):
        x = self.entity_embedding(entity)
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_norm))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x = self.conv2(x, edge_index, edge_type, edge_norm)
        
        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))


class RGAT(torch.nn.Module):
    def __init__(self, num_entities, num_relations, dropout):
        super(RGAT, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        
        self.conv1 = RGATConv(100, 100, num_relations * 2, num_bases=4)
        self.conv2 = RGATConv(100, 100, num_relations * 2, num_bases=4)
        # emb_matrix = torch.eye(2*num_relations)
        # self.one_hot_embedding = torch.nn.Embedding.from_pretrained(emb_matrix, freeze = True)

        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type, edge_norm):
        x = self.entity_embedding(entity)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)

        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))


