import torch
import json
import joblib
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from gatv2 import GATv2EdgePredictor

# === File paths ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
test_data_dir = os.path.join(project_root, "data", "data_splits", "test_data_monthly")
scaler_path = os.path.join(project_root, "src", "utils", "data_splits", "scalers", "2021_to_2023", "target_scaler.pkl")
config_path = os.path.join(project_root, "src", "models", "gat", "hpo_models", "best_config_overall.json")
model_path = os.path.join(project_root, "src", "models", "gat", "hpo_models", "best_model_overall.pth")

# === Load configuration and scaler ===
with open(config_path, "r") as f:
    config = json.load(f)

y_scaler = joblib.load(scaler_path)

# === Training helper ===
def train_model(model, data, val_data, epochs=3, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_data)
            val_loss = loss_fn(val_out, val_data.y)
        print(f"  Epoch {epoch+1}/{epochs} - Train loss: {loss.item():.4f}, Val loss: {val_loss.item():.4f}")

# === Load one dummy month to initialize model ===
dummy_data = torch.load(os.path.join(test_data_dir, "month_01.pt"), weights_only=False)
model = GATv2EdgePredictor(
    in_channels=dummy_data.num_node_features,
    hidden_channels=config["hidden_channels"],
    out_channels=config["out_channels"],
    edge_dim=dummy_data.edge_attr.shape[1],
    heads=config["heads"]
)

# === Load pretrained weights once (initial state for January) ===
model.load_state_dict(torch.load(model_path))

# === Iterate over months 2 to 12 (fine-tune on month-1, evaluate on month) ===
total_squared_error = 0
total_edges = 0

print("\n=== Monthly RMSEs with Cumulative Fine-Tuning ===")
for month in range(2, 13):
    # Load previous month for fine-tuning
    train_month_data = torch.load(os.path.join(test_data_dir, f"month_{month-1:02d}.pt"), weights_only=False)

    # Split into train and val (e.g., 80/20 split of edges)
    num_edges = train_month_data.edge_index.shape[1]
    indices = np.arange(num_edges)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    def extract_subgraph(data, indices):
        new_data = data.clone()
        new_data.edge_index = data.edge_index[:, indices]
        new_data.edge_attr = data.edge_attr[indices]
        new_data.y = data.y[indices]
        return new_data

    train_data = extract_subgraph(train_month_data, train_idx)
    val_data = extract_subgraph(train_month_data, val_idx)

    # Fine-tune the model on (month-1)
    model.train()
    train_model(model, train_data, val_data, epochs=100, lr=0.001)

    # === Evaluate on current month ===
    model.eval()
    eval_data = torch.load(os.path.join(test_data_dir, f"month_{month:02d}.pt"), weights_only=False)

    with torch.no_grad():
        predictions = model(eval_data)

    pred_orig = y_scaler.inverse_transform(predictions.cpu().numpy())
    true_orig = y_scaler.inverse_transform(eval_data.y.cpu().numpy())

    mse = mean_squared_error(true_orig, pred_orig)
    rmse = np.sqrt(mse)
    edges = eval_data.edge_index.size(1)

    print(f"Month {month:02d}: RMSE = {rmse:.2f} (Edges: {edges})")

    total_squared_error += mse * edges
    total_edges += edges

# === Overall weighted RMSE ===
overall_rmse = np.sqrt(total_squared_error / total_edges)
print(f"\n===> Overall weighted RMSE (cumulative fine-tuning): {overall_rmse:.2f}")
