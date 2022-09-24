from tokenize import Funny
import torch
from torch_geometric.nn import RGCNConv, GATConv
import torch.nn.functional as F
from torch.nn import Parameter


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
    