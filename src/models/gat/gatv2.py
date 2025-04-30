import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv

class GATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, heads=1):
        super(GATv2, self).__init__()

        # First GATv2 layer, with edge attributes
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim)

        # Second GATv2 layer, output dimension = out_channels
        self.gat2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        # Apply first GATv2 layer with edge attributes
        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)

        # Apply second GATv2 layer
        x = self.gat2(x, edge_index, edge_attr)
        return x

class GATv2EdgePredictor(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 edge_dim, 
                 heads=1):
        super(GATv2EdgePredictor, self).__init__()

        # 1. GATv2 model for computing node embeddings
        self.gnn = GATv2(in_channels, hidden_channels, out_channels, edge_dim, heads)

        # 2. Edge MLP to predict edge attributes (e.g., "tracks")
        self.edge_mlp = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)  # Output: a single scalar per edge
        )

    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Data object with attributes:
                  - x: node features
                  - edge_index: edge connectivity (COO format)
                  - edge_attr: edge attributes

        Returns:
            pred: Tensor of shape [num_edges, 1] with predicted edge weights (e.g., "tracks")
        """
        # Compute node embeddings using the GATv2 model
        x = self.gnn(data.x, data.edge_index, data.edge_attr)  # [num_nodes, out_channels]

        # Construct edge representations by concatenating source and target node embeddings
        row, col = data.edge_index  # source & target node indices for each edge
        edge_inputs = torch.cat([x[row], x[col]], dim=1)  # [num_edges, out_channels * 2]

        # Predict edge weights
        pred = self.edge_mlp(edge_inputs)  # [num_edges, 1]
        return pred
