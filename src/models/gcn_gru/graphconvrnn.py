# graphconvrnn.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GraphConvRNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, rnn_layers=1):
        super().__init__()
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.gru = nn.GRU(hidden_channels, hidden_channels, rnn_layers, batch_first=True)
        self.edge_mlp = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None, hidden=None):
        x = self.gcn(x, edge_index)
        x = torch.relu(x)
        x_seq = x.unsqueeze(0)
        out_seq, hidden = self.gru(x_seq, hidden)
        node_embeddings = out_seq.squeeze(0)
        source, target = edge_index
        edge_embeddings = (node_embeddings[source] + node_embeddings[target]) / 2
        edge_output = self.edge_mlp(edge_embeddings)
        return edge_output.squeeze(), hidden
