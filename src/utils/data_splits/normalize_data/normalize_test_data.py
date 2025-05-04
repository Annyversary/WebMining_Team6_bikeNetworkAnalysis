import numpy as np
import os
import joblib
import torch

def normalize_test_features(test_batch):
    """
    Normalizes a batched test graph using pre-fitted scalers from training phase.
    """

    # === Load scalers === #
    scaler_dir = os.path.join(os.path.dirname(__file__), "..", "scalers", "2021_to_2023")
    scaler_dir = os.path.abspath(scaler_dir)
    y_scaler = joblib.load(os.path.join(scaler_dir, "target_scaler.pkl"))
    feat_scaler = joblib.load(os.path.join(scaler_dir, "edge_scaler.pkl"))
    node_scaler = joblib.load(os.path.join(scaler_dir, "node_scaler.pkl"))

    # === Define indices === #
    feat_indices = [0, 2]  # id, speed_rel
    month_idx = 1  # month is at index 1
    target_idx = 4

    # --- Target y --- #
    print("Before normalization (target y):", test_batch.edge_attr[:, target_idx].numpy()[:10])  # Debugging: before normalization
    test_batch.y = test_batch.edge_attr[:, target_idx]
    test_batch.y = torch.tensor(
        y_scaler.transform(test_batch.y.view(-1, 1)), dtype=torch.float32
    )
    print("After normalization (target y):", test_batch.y.numpy()[:10])

    # --- Edge features --- #
    edge_feats = test_batch.edge_attr[:, feat_indices].numpy()
    print("Before normalization (edge features):", edge_feats[:10])
    edge_feats_scaled = torch.tensor(
        feat_scaler.transform(edge_feats), dtype=torch.float32
    )
    print("After normalization (edge features):", edge_feats_scaled.numpy()[:10])

    # Month cyclical encoding
    month_raw = test_batch.edge_attr[:, month_idx]
    month_sin = torch.sin(2 * np.pi * month_raw / 12).view(-1, 1)
    month_cos = torch.cos(2 * np.pi * month_raw / 12).view(-1, 1)

    test_batch.edge_attr = torch.cat([edge_feats_scaled, month_sin, month_cos], dim=1)

    # --- Node features --- #
    print("Before normalization (node features):", test_batch.x.numpy()[:10])
    test_batch.x = torch.tensor(
        node_scaler.transform(test_batch.x.numpy()), dtype=torch.float32
    )
    print("After normalization (node features):", test_batch.x.numpy()[:10])

    return test_batch
