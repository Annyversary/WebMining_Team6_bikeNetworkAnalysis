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

    print(f"[INFO] Loaded scalers from: {scaler_dir}")

    # === Define indices === #
    feat_indices = [3, 1]  # year and speed_rel
    month_idx = 2  # month
    target_idx = 4

    # === Target y === #
    print(f"[DEBUG] test_batch.edge_attr shape: {test_batch.edge_attr.shape}")
    print("First 5 edge_attr rows (raw):", test_batch.edge_attr[:5])

    print("\n[DEBUG] Target y BEFORE normalization:")
    print(test_batch.edge_attr[:, target_idx][:10].numpy())

    test_batch.y = test_batch.edge_attr[:, target_idx]
    test_batch.y = torch.tensor(
        y_scaler.transform(test_batch.y.view(-1, 1)),
        dtype=torch.float32
    )

    print("[DEBUG] Target y AFTER normalization:")
    print(test_batch.y[:10].numpy())

    # === Edge features === #
    edge_feats = test_batch.edge_attr[:, feat_indices].numpy()
    print("\n[DEBUG] Edge features BEFORE normalization (selected columns):")
    print(edge_feats[:5])

    edge_feats_scaled = torch.tensor(
        feat_scaler.transform(edge_feats), dtype=torch.float32
    )
    print("[DEBUG] Edge features AFTER normalization:")
    print(edge_feats_scaled[:5].numpy())

    # === Month cyclical encoding === #
    month_raw = test_batch.edge_attr[:, month_idx]
    print("\n[DEBUG] Raw months:", month_raw[:10])
    month_sin = torch.sin(2 * np.pi * month_raw / 12).view(-1, 1)
    month_cos = torch.cos(2 * np.pi * month_raw / 12).view(-1, 1)

    print("[DEBUG] Month sine values:", month_sin[:5].squeeze().numpy())
    print("[DEBUG] Month cosine values:", month_cos[:5].squeeze().numpy())

    # === New edge_attr === #
    test_batch.edge_attr = torch.cat([edge_feats_scaled, month_sin, month_cos], dim=1)
    print("\n[DEBUG] edge_attr AFTER recombination (scaled + month sin/cos):")
    print(test_batch.edge_attr[:5])

    # === Node features === #
    print("\n[DEBUG] Node features BEFORE normalization:")
    print(test_batch.x[:5].numpy())

    test_batch.x = torch.tensor(
        node_scaler.transform(test_batch.x.numpy()), dtype=torch.float32
    )

    print("[DEBUG] Node features AFTER normalization:")
    print(test_batch.x[:5].numpy())

    return test_batch
