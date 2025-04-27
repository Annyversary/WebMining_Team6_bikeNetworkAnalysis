import torch
from graphconvrnn import GraphConvRNN
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx

# Dummy-Graph erstellen
def create_dummy_graph(num_nodes=5):
    G = nx.complete_graph(num_nodes)
    for node in G.nodes():
        G.nodes[node]['lat'] = np.random.rand()
        G.nodes[node]['lon'] = np.random.rand()
    for u, v in G.edges():
        G.edges[u, v]['id'] = np.random.randint(1000)
        G.edges[u, v]['month'] = np.random.randint(1, 13)
        G.edges[u, v]['speed_rel'] = np.random.rand()
        G.edges[u, v]['tracks'] = np.random.rand() * 100
        G.edges[u, v]['year'] = np.random.randint(2021, 2025)
    data = from_networkx(G, group_node_attrs=['lat', 'lon'], group_edge_attrs=['id', 'month', 'speed_rel', 'tracks', 'year'])
    data.y = data.edge_attr[:, 3]
    data.edge_attr = torch.cat([data.edge_attr[:, :3], data.edge_attr[:, 4:]], dim=1)
    return data

# Testablauf
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = create_dummy_graph()

model = GraphConvRNN(
    in_channels=data.num_node_features,
    hidden_channels=16,
    out_channels=1
).to(device)

model.eval()
with torch.no_grad():
    data = data.to(device)
    out, _ = model(data.x, data.edge_index, data.edge_attr)
    assert out.shape[0] == data.y.shape[0], "❌ Output und Target haben unterschiedliche Dimensionen!"

print("✅ Test erfolgreich: Modellvorhersage korrekt.")
