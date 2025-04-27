import torch
from torch_geometric.utils import from_networkx

def transform_networkx_into_pyg(G):
    """
    Converts a NetworkX graph to a PyTorch Geometric Data object.
    All node and edge attributes are preserved and added as node features and edge features, respectively.

    Args:
        G (networkx.Graph): The input NetworkX graph, which should contain all node and edge attributes.

    Returns:
        torch_geometric.data.Data: A PyTorch Geometric Data object containing the graph's node features, 
        edge indices, and edge attributes.
    """

    # Convert the NetworkX graph to a PyTorch Geometric Data object
    data = from_networkx(G)
    
    # --- 1. Add node features (e.g., lon, lat) to data.x ---
    node_features = []
    for node, attrs in G.nodes(data=True):
        lon = float(attrs.get("lon", 0.0)) 
        lat = float(attrs.get("lat", 0.0))
        node_features.append([lon, lat])  

    data.x = torch.tensor(node_features, dtype=torch.float) 
    
    # --- 2. Edge attribute handling
    edge_keys = list(next(iter(G.edges(data=True)))[2].keys())

    edge_features = []
    for key in edge_keys:
        dtype = torch.float32 if key == 'speed_rel' else torch.long
        edge_features.append(torch.tensor([G.edges[u, v][key] for u, v in G.edges()], dtype=dtype).unsqueeze(1))

    data.edge_attr = torch.cat(edge_features, dim=1)

    return data
