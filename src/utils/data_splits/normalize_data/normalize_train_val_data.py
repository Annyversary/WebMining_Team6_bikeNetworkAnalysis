import numpy as np
import joblib
import torch
import os

from sklearn.preprocessing import StandardScaler

def normalize_feature_mixedSplit(train_batch, val_batch):
    """
    Normalizes edge and node attributes in batched train/val graph data.
    Applies cyclical encoding for 'month' in edges and uses StandardScaler for all other features.
    Also saves the fitted scalers to disk for later reuse.
    """

    print("\nStart Normalization (mixed split)")
    
    ### === Normalize targets (y) === ###
    y_train = train_batch.y.view(-1, 1).numpy()
    print(f"Before target scaling: y_train stats: mean={y_train.mean():.4f}, std={y_train.std():.4f}, min={y_train.min()}, max={y_train.max()}")

    y_scaler = StandardScaler().fit(y_train)

    train_batch.y = torch.tensor(y_scaler.transform(train_batch.y.view(-1, 1)), dtype=torch.float32)
    val_batch.y = torch.tensor(y_scaler.transform(val_batch.y.view(-1, 1)), dtype=torch.float32)

    print(f"After target scaling: train y mean={train_batch.y.mean():.4f}, std={train_batch.y.std():.4f}")
    print(f"After target scaling: val   y mean={val_batch.y.mean():.4f}, std={val_batch.y.std():.4f}")

    ### === Normalize edge attributes === ###
    month_idx = 2
    feat_indices = [3, 1]  #speed_rel and year

    print(f"Edge attr shape before scaling: {train_batch.edge_attr.shape}")
    print(f"Feature indices used for scaling: {feat_indices} (i.e., speed_rel, year)")

    print("\nFirst 5 edge_attr rows BEFORE scaling:\n", train_batch.edge_attr[:5])


    edge_feats_train = train_batch.edge_attr[:, feat_indices].numpy()
    print(f"Edge feature stats before scaling:\n"
          f"speed_rel mean={edge_feats_train[:, 0].mean():.4f}, year mean={edge_feats_train[:, 1].mean():.4f}")

    feat_scaler = StandardScaler().fit(edge_feats_train)

    for label, batch in [("train", train_batch), ("val", val_batch)]:
        feat_tensor = batch.edge_attr[:, feat_indices]
        feat_scaled = torch.tensor(feat_scaler.transform(feat_tensor.numpy()), dtype=torch.float32)

        # Cyclical encoding for month
        month_raw = batch.edge_attr[:, month_idx]
        month_sin = torch.sin(2 * np.pi * month_raw / 12).view(-1, 1)
        month_cos = torch.cos(2 * np.pi * month_raw / 12).view(-1, 1)

        batch.edge_attr = torch.cat([feat_scaled, month_sin, month_cos], dim=1)

        print(f"{label} edge_attr shape after scaling and month encoding: {batch.edge_attr.shape}")
        print(f"{label} edge_attr sample[0]: {batch.edge_attr[0]}")

    ### === Normalize node features (lon, lat) === ###
    node_feats_train = train_batch.x.numpy()
    print(f"Node features shape: {train_batch.x.shape}")
    print(f"Before node scaling: lon mean={node_feats_train[:, 0].mean():.6f}, lat mean={node_feats_train[:, 1].mean():.6f}")

    node_scaler = StandardScaler().fit(node_feats_train)

    train_batch.x = torch.tensor(node_scaler.transform(train_batch.x.numpy()), dtype=torch.float32)
    val_batch.x = torch.tensor(node_scaler.transform(val_batch.x.numpy()), dtype=torch.float32)

    print(f"After node scaling: train x mean={train_batch.x.mean():.4f}, std={train_batch.x.std():.4f}")

    ### === Save scalers === ###
    scaler_dir = os.path.join("src", "utils", "data_splits", "scalers", "2021_to_2023")
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(y_scaler, os.path.join(scaler_dir, "target_scaler.pkl"))
    joblib.dump(feat_scaler, os.path.join(scaler_dir, "edge_scaler.pkl"))
    joblib.dump(node_scaler, os.path.join(scaler_dir, "node_scaler.pkl"))

    print("Normalization complete. Scalers saved.")
    return train_batch, val_batch


def normalize_feature_timedSplit(train_batch, val_batch):
    """
    Normalizes edge and node attributes in batched train/val graph data.
    Applies cyclical encoding for 'month' in edges and uses StandardScaler for all other features.
    Also saves the fitted scalers to disk for later reuse.
    """
    
    ### === Normalize targets (y) === ###
    y_train = train_batch.y.view(-1, 1).numpy()
    y_scaler = StandardScaler().fit(y_train)

    train_batch.y = torch.tensor(y_scaler.transform(train_batch.y.view(-1, 1)), dtype=torch.float32)
    val_batch.y = torch.tensor(y_scaler.transform(val_batch.y.view(-1, 1)), dtype=torch.float32)

    ### === Normalize edge attributes === ###
    month_idx = 2
    feat_indices = [3, 1]  #speed_rel and year

    edge_feats_train = train_batch.edge_attr[:, feat_indices].numpy()
    feat_scaler = StandardScaler().fit(edge_feats_train)

    for batch in [train_batch, val_batch]:
        feat_tensor = batch.edge_attr[:, feat_indices]
        feat_scaled = torch.tensor(feat_scaler.transform(feat_tensor.numpy()), dtype=torch.float32)

        # Cyclical encoding for month
        month_raw = batch.edge_attr[:, month_idx]
        month_sin = torch.sin(2 * np.pi * month_raw / 12).view(-1, 1)
        month_cos = torch.cos(2 * np.pi * month_raw / 12).view(-1, 1)

        batch.edge_attr = torch.cat([feat_scaled, month_sin, month_cos], dim=1)

    ### === Normalize node features (lon, lat) === ###
    node_feats_train = train_batch.x.numpy()
    node_scaler = StandardScaler().fit(node_feats_train)

    train_batch.x = torch.tensor(node_scaler.transform(train_batch.x.numpy()), dtype=torch.float32)
    val_batch.x = torch.tensor(node_scaler.transform(val_batch.x.numpy()), dtype=torch.float32)

    ### === Save scalers === ###
    scaler_dir = os.path.join("src", "utils", "data_splits", "scalers", "timned")
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(y_scaler, os.path.join(scaler_dir, "target_scaler.pkl"))
    joblib.dump(feat_scaler, os.path.join(scaler_dir, "edge_scaler.pkl"))
    joblib.dump(node_scaler, os.path.join(scaler_dir, "node_scaler.pkl"))

    return train_batch, val_batch