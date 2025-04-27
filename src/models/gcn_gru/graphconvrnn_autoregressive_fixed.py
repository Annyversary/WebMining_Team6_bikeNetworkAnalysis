import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader, Batch
import networkx as nx
from torch_geometric.utils import from_networkx
import glob
import os
import pandas as pd
import json
from tsl.metrics.numpy.functional import rmse, mape
import numpy as np


# --- EarlyStopping Helper Class ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# --- Model Definition ---
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


# --- Data Loading ---
def load_graphml(year):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    path = os.path.join(PROJECT_ROOT, "data", "graphml", str(year), "*.graphml")
    print(f"[INFO] Looking for GraphML files at: {path}")

    files = glob.glob(path)
    print(f"[INFO] Found {len(files)} GraphML files for year {year}")

    if not files:
        raise FileNotFoundError(f"[ERROR] No GraphML files found for year {year} at {path}")

    data_list = []
    for f in sorted(files):
        print(f"[INFO] Loading graph: {f}")
        try:
            G = nx.read_graphml(f)
            data = from_networkx(G, group_node_attrs=['lat', 'lon'],
                                 group_edge_attrs=['id', 'month', 'speed_rel', 'tracks', 'year'])
            data.y = data.edge_attr[:, 3]  # 'tracks' wird Prediction Target
            data.edge_attr = torch.cat([data.edge_attr[:, :3], data.edge_attr[:, 4:]],
                                       dim=1)  # 'tracks' wird aus edge_attr entfernt
            data_list.append(data)
        except Exception as e:
            print(f"[ERROR] Failed to load graph {f}: {e}")

    return data_list

train_list = load_graphml(2021) + load_graphml(2022)
val_list = load_graphml(2023)
test_list = load_graphml(2024)

# Nach dem Laden der Daten
print(f"Loaded training graphs: {len(train_list)}")
print(f"Loaded validation graphs: {len(val_list)}")

if len(train_list) == 0:
    raise ValueError("No training data available! Please check data loading and splitting.")

train_loader = DataLoader(train_list, batch_size=1, shuffle=True)
val_loader = DataLoader(val_list, batch_size=1, shuffle=False)

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Metric Functions ---
def calculate_rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def calculate_mape(y_true, y_pred):
    epsilon = 1e-8
    return (torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100).item()


# --- Hyperparameter Search with Early Stopping ---
best_val_loss = float('inf')
best_val_rmse = float('inf')
best_val_mape = float('inf')
best_params = {}

for lr in [1e-3, 5e-4]:
    for wd in [0, 1e-5]:
        model = GraphConvRNN(
            in_channels=train_list[0].num_node_features,
            hidden_channels=64,
            out_channels=1
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        early_stopper = EarlyStopping(patience=5)

        for epoch in range(1, 51):
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out, _ = model(data.x, data.edge_index, data.edge_attr)
                loss = nn.MSELoss()(out.view(-1), data.y.view(-1))
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            y_true_all = []
            y_pred_all = []

            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out, _ = model(data.x, data.edge_index, data.edge_attr)
                    y_true_all.append(data.y.view(-1).cpu().numpy())
                    y_pred_all.append(out.view(-1).cpu().numpy())
                    val_loss += nn.MSELoss()(out.view(-1), data.y.view(-1)).item()

            val_loss /= len(val_loader)

            if early_stopper.step(val_loss):
                print(f"Early stopping during hyperparameter search at epoch {epoch}")
                break

            y_true_all = np.concatenate(y_true_all)
            y_pred_all = np.concatenate(y_pred_all)
            val_rmse = rmse(y_pred_all, y_true_all)
            val_mape = mape(y_pred_all, y_true_all)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_rmse = val_rmse
                best_val_mape = val_mape
                best_params = {
                    'lr': lr,
                    'weight_decay': wd,
                    'val_loss': best_val_loss,
                    'val_rmse': best_val_rmse,
                    'val_mape': best_val_mape
                }

with open('best_hyperparams.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print('Beste Hyperparameter gefunden:')
print(json.dumps(best_params, indent=4))

# --- Final Training with Early Stopping ---
model = GraphConvRNN(
    in_channels=train_list[0].num_node_features,
    hidden_channels=64,
    out_channels=1
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
early_stopper_final = EarlyStopping(patience=10)

for epoch in range(1, 301):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index, data.edge_attr)
        loss = nn.MSELoss()(out.view(-1), data.y.view(-1))
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.edge_attr)
            val_loss += nn.MSELoss()(out.view(-1), data.y.view(-1)).item()

    val_loss /= len(val_loader)

    if early_stopper_final.step(val_loss):
        print(f"Early stopping final training at epoch {epoch}")
        break

# --- Forecasting ---
current_graph = val_list[-1].clone().to(device)
hidden = None
forecasts = []

for i in range(12):
    model.eval()
    with torch.no_grad():
        out, hidden = model(current_graph.x, current_graph.edge_index, current_graph.edge_attr, hidden)
    forecasts.append(out.cpu())
    new_edge_attr = torch.cat([current_graph.edge_attr.cpu(), out.unsqueeze(1)], dim=1)
    new_edge_attr = torch.cat([new_edge_attr[:, :3], new_edge_attr[:, 4:]], dim=1)
    current_graph.edge_attr = new_edge_attr.to(device)
    current_graph.y = out.to(device)

# --- Save Forecasts ---
df = pd.DataFrame([f.numpy() for f in forecasts]).T
df.columns = [f'Month_{i + 1}' for i in range(12)]
df.to_csv('forecasts_2024.csv', index=False)
print('Forecasts gespeichert als forecasts_2024.csv')



