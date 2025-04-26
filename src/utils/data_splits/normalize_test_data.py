import numpy as np
import joblib
import torch
import os

def normalize_test_features(test_batch):
    """
    Normalizes a batched test graph using pre-fitted scalers from training phase.
    
    This includes:
      - Standardizing the target variable (from edge_attr),
      - Applying cyclical encoding to the 'month' edge feature,
      - Normalizing other edge features (e.g., speed_rel, year),
      - Normalizing node features (coordinates).

    Parameters:
    -----------
    test_batch : torch_geometric.data.Batch
        A batched PyTorch Geometric Data object representing test graphs.

    Returns:
    --------
    test_batch : torch_geometric.data.Batch
        The normalized batch.
    """
    
    # === Load scalers === #
    scaler_dir = os.path.join("src","utils","data_splits","scalers")
    y_scaler = joblib.load(os.path.join(scaler_dir, "target_scaler.pkl"))
    feat_scaler = joblib.load(os.path.join(scaler_dir, "edge_scaler.pkl"))
    node_scaler = joblib.load(os.path.join(scaler_dir, "node_scaler.pkl"))

    # === Define indices === #
    target_idx = 4
    feat_indices = [0, 2]
    month_idx = 1

    # --- Target y --- #
    print("Before normalization (target y):", test_batch.edge_attr[:, target_idx].numpy()[:10])  # Debugging: before normalization
    test_batch.y = test_batch.edge_attr[:, target_idx]
    test_batch.y = torch.tensor(
        y_scaler.transform(test_batch.y.view(-1, 1)), dtype=torch.float32
    )
    print("After normalization (target y):", test_batch.y.numpy()[:10])  # Debugging: after normalization

    # --- Edge features --- #
    edge_feats = test_batch.edge_attr[:, feat_indices].numpy()
    print("Before normalization (edge features):", edge_feats[:10])  # Debugging: before normalization
    edge_feats_scaled = torch.tensor(
        feat_scaler.transform(edge_feats), dtype=torch.float32
    )
    print("After normalization (edge features):", edge_feats_scaled.numpy()[:10])  # Debugging: after normalization

    # Month cyclical encoding
    month_raw = test_batch.edge_attr[:, month_idx]
    month_sin = torch.sin(2 * np.pi * month_raw / 12).view(-1, 1)
    month_cos = torch.cos(2 * np.pi * month_raw / 12).view(-1, 1)

    test_batch.edge_attr = torch.cat([edge_feats_scaled, month_sin, month_cos], dim=1)

    # --- Node features --- #
    print("Before normalization (node features):", test_batch.x.numpy()[:10])  # Debugging: before normalization
    test_batch.x = torch.tensor(
        node_scaler.transform(test_batch.x.numpy()), dtype=torch.float32
    )
    print("After normalization (node features):", test_batch.x.numpy()[:10])  # Debugging: after normalization

    return test_batch
