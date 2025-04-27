import os
import os
import glob
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Batch
from src.utils.data_splits.normalize_data import normalize_train_val_data, normalize_test_data
from src.utils.data_splits.helper_functions import print_datasplit_info as print_info

def load_graphml_files_for_year(year):
    base_path = os.path.join("..", "..", "..", "data", "graphml", str(year))
    graph_files = glob.glob(os.path.join(base_path, "*.graphml"))

    if not graph_files:
        print(f"[WARN] No GraphML files found for year {year} in {base_path}")

    data_list = []
    for filepath in graph_files:
        try:
            G = nx.read_graphml(filepath)

            node_attrs = ['lat', 'lon']
            edge_attrs = ['id', 'month', 'speed_rel', 'tracks', 'year']

            data = from_networkx(
                G,
                group_node_attrs=node_attrs,
                group_edge_attrs=edge_attrs
            )

            if not hasattr(data, 'edge_attr') or data.edge_attr is None:
                print(f"[WARN] Graph {filepath} loaded without edge_attr!")

            data_list.append(data)

        except Exception as e:
            print(f"[ERROR] Failed to load {filepath}: {e}")

    print(f"[INFO] Loaded {len(data_list)} graphs for year {year}")
    return data_list

def prepare_data(data_list, edge_attr_key_index=4):
    """Assigns y from edge_attr and removes y-column from edge_attr."""
    prepared_list = []

    for i, data in enumerate(data_list):
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        if edge_attr is None or edge_attr.size(1) <= edge_attr_key_index:
            print(f"[WARN] Graph {i} has insufficient edge attributes.")
            continue

        data.y = edge_attr[:, edge_attr_key_index]
        data.edge_attr = torch.cat([
            edge_attr[:, :edge_attr_key_index],
            edge_attr[:, edge_attr_key_index+1:]
        ], dim=1)

        prepared_list.append(data)

    return prepared_list

def main():
    save_dir = os.path.join("..", "..", "..", "data", "data_splits")
    os.makedirs(save_dir, exist_ok=True)

    train_save_path = os.path.join(save_dir, "train_data.pt")
    val_save_path = os.path.join(save_dir, "val_data.pt")
    test_save_path = os.path.join(save_dir, "test_data.pt")

    # Load data
    train_data_list = load_graphml_files_for_year(2021) + load_graphml_files_for_year(2022)
    val_data_list = load_graphml_files_for_year(2023)
    test_data_list = load_graphml_files_for_year(2024)

    # Check if lists are not empty
    if not train_data_list:
        raise ValueError("No training data was loaded!")
    if not val_data_list:
        raise ValueError("No validation data was loaded!")
    if not test_data_list:
        raise ValueError("No test data was loaded!")

    # Prepare data (set y, remove y-column from edge_attr)
    train_data_list = prepare_data(train_data_list)
    val_data_list = prepare_data(val_data_list)
    test_data_list = prepare_data(test_data_list)


    train_batch = Batch.from_data_list(train_data_list)
    val_batch = Batch.from_data_list(val_data_list)
    test_batch = Batch.from_data_list(test_data_list)

    # Normalize train and val together
    train_batch, val_batch = normalize_train_val_data.normalize_feature(train_batch, val_batch)

    # Normalize test set separately (using train scaler)
    test_batch = normalize_test_data.normalize_test_features(test_batch)

    # Print stats
    print("\nTrain Data Statistics:")
    print_info.print_batch_shape(train_batch)
    print("\nValidation Data Statistics:")
    print_info.print_batch_shape(val_batch)
    print("\nTest Data Statistics:")
    print_info.print_batch_shape(test_batch)

    # Save
    torch.save(train_batch, train_save_path)
    torch.save(val_batch, val_save_path)
    torch.save(test_batch, test_save_path)

    print(f"\nTrain data saved to: {train_save_path}")
    print(f"Validation data saved to: {val_save_path}")
    print(f"Test data saved to: {test_save_path}")

if __name__ == "__main__":
    main()

