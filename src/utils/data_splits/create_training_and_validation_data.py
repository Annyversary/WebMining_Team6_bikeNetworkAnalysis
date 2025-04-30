import os
import torch

from torch_geometric.data import Batch
from normalize_data.normalize_train_val_data import normalize_feature
from helper_functions.load_graphml_files import load_graphml_files
from helper_functions.print_datasplit_info import print_batch_shape

def split_train_val(data_list, val_ratio=0.2, seed=42, edge_attr_key_index=4):
    """
    Splits a list of PyTorch Geometric Data objects into training and validation sets
    for edge regression tasks.

    Parameters:
    -----------
    data_list : list of torch_geometric.data.Data
        List of graphs to be split into training and validation sets.

    val_ratio : float, optional (default=0.2)
        Proportion of edges to be used for validation in each graph.

    seed : int, optional (default=42)
        Random seed for reproducibility.

    edge_attr_key_index : int, optional (default=4)
        The index of the edge attribute that should be predicted (e.g., 'tracks').
        This attribute will be used as the target (`y`) for the regression task.

    Returns:
    --------
    train_data, val_data : torch_geometric.data.Batch
        Batched training and validation data containing the graphs' edge indices,
        edge attributes, and the target edge attribute (`y`) for regression.
    """

    torch.manual_seed(seed)

    train_list, val_list = [], []
    total_train_edges = 0
    total_val_edges = 0

    for i, data in enumerate(data_list):
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        num_edges = edge_index.size(1)
        num_val = int(val_ratio * num_edges)
        perm = torch.randperm(num_edges)

        val_idx = perm[:num_val]
        train_idx = perm[num_val:]

        # Training Data
        train_data = data.clone()
        train_data.edge_index = edge_index[:, train_idx]
        train_data.edge_attr = torch.cat([edge_attr[train_idx][:, :edge_attr_key_index], edge_attr[train_idx][:, edge_attr_key_index+1:]], dim=1)
        train_data.y = edge_attr[train_idx][:, edge_attr_key_index]

        # Validation Data
        val_data = data.clone()
        val_data.edge_index = edge_index[:, val_idx]
        val_data.edge_attr = torch.cat([edge_attr[val_idx][:, :edge_attr_key_index], edge_attr[val_idx][:, edge_attr_key_index+1:]], dim=1)
        val_data.y = edge_attr[val_idx][:, edge_attr_key_index]

        train_list.append(train_data)
        val_list.append(val_data)

        total_train_edges += train_data.edge_index.size(1)
        total_val_edges += val_data.edge_index.size(1)

        print(f"Graph {i}: Train edges = {train_data.edge_index.size(1)}, Val edges = {val_data.edge_index.size(1)}")

    # Batch the split data
    train_data = Batch.from_data_list(train_list)
    val_data = Batch.from_data_list(val_list)

    print(f"\nTotal train edges (batched): {total_train_edges}")
    print(f"Total val edges   (batched): {total_val_edges}")

    return train_data, val_data


def main(years=[2021, 2022, 2023], val_ratio=0.2):
    """
    Main pipeline for loading graph data, preprocessing it, and splitting into train/val sets.

    Parameters:
    -----------
    years : list of int, optional (default=[2021, 2022, 2023])
        The years for which GraphML files will be loaded.

    val_ratio : float, optional (default=0.2)
        Proportion of edges to be used for validation during the train/validation split.

    Returns:
    --------
    None
    """

    save_dir = os.path.join("data", "data_splits", "2021_to_2023")
    os.makedirs(save_dir, exist_ok=True)
    train_save_path = os.path.join(save_dir, "train_data.pt")
    val_save_path = os.path.join(save_dir, "val_data.pt")

    # Load GraphML files for the specified years and convert to PyTorch Geometric Data objects
    data_list = load_graphml_files(years)

    # Split data into train and validation sets
    train_data, val_data = split_train_val(data_list, val_ratio=val_ratio)

    # Normalize features in the training and validation data
    train_data, val_data = normalize_feature(train_data, val_data)

    print("\nTrain Data Statistics:")
    print_batch_shape(train_data)
    print("\nValidation Data Statistics:")
    print_batch_shape(val_data)

    # Save the train and validation data
    torch.save(train_data, train_save_path)
    torch.save(val_data, val_save_path)

    print(f"\nTrain data saved to: {train_save_path}")
    print(f"Val data saved to: {val_save_path}")

if __name__ == "__main__":
    main()