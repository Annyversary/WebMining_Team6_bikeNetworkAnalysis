import torch
import json
import joblib
import os
import sys
import import_ipynb

src_path = os.path.abspath(os.path.join(os.getcwd(), "..",  ".."))
if src_path not in sys.path:
    sys.path.append(src_path)

from torch_geometric.data import Data
from models.gat.gatv2 import GATv2EdgePredictor


# === Dateipfade ===
config_path = "hpo_models/best_config_overall.json"
model_path = "hpo_models/best_model_overall.pth"
test_data_path = "../../../data/data_splits/test_data.pt"
scaler_path = "../../utils/data_splits/scalers/target_scaler.pkl"

# === 1. Lade Testdaten ===
test_data = torch.load(test_data_path, weights_only=False)

# === 2. Lade die beste Konfiguration ===
with open(config_path, "r") as f:
    config = json.load(f)

# === 3. Initialisiere Modell gemäß Konfiguration ===
model = GATv2EdgePredictor(
    in_channels=test_data.num_node_features,
    hidden_channels=config["hidden_channels"],
    out_channels=config["out_channels"],
    edge_dim=test_data.edge_attr.shape[1],
    heads=config["heads"]
)

# === 4. Lade trainierte Gewichte ===
model.load_state_dict(torch.load(model_path))
model.eval()

# === 5. Mache Vorhersagen ===
with torch.no_grad():
    predictions = model(test_data)

# === 6. Lade den Skaler, um Werte zurückzutransformieren ===
y_scaler = joblib.load(scaler_path)
pred_orig = y_scaler.inverse_transform(predictions.cpu().numpy())
true_orig = y_scaler.inverse_transform(test_data.y.cpu().numpy())

# === 7. RMSE berechnen ===
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(true_orig, pred_orig))
print(f"RMSE on test set (original scale): {rmse:.2f}")

print("True values (first 10):", true_orig[:10].flatten())
print("Predicted values (first 10):", pred_orig[:10].flatten())

test_data = torch.load("../../../data/data_splits/test_data.pt", weights_only=False)
print("y[:10] (scaled):", test_data.y[:10].flatten())
print("y max (scaled):", torch.max(test_data.y))

y_scaler = joblib.load("../../utils/data_splits/scalers/target_scaler.pkl")
y_unscaled = y_scaler.inverse_transform(test_data.y.cpu().numpy())
print("y[:10] (original scale):", y_unscaled[:10].flatten())
print("Max y (original scale):", np.max(y_unscaled))

print("Target scaler stats:")
print("  Mean:", y_scaler.mean_ if hasattr(y_scaler, 'mean_') else "N/A")
print("  Std:", y_scaler.scale_ if hasattr(y_scaler, 'scale_') else "N/A")
print("  Min:", y_scaler.data_min_ if hasattr(y_scaler, 'data_min_') else "N/A")
print("  Max:", y_scaler.data_max_ if hasattr(y_scaler, 'data_max_') else "N/A")


