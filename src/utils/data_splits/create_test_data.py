import os
import torch

from torch_geometric.data import Batch
from helper_functions.load_graphml_files import load_graphml_files
from normalize_data.normalize_test_data import normalize_test_features
from helper_functions.print_datasplit_info import print_batch_shape

def main(years=[2024]):
    save_dir = os.path.join("data", "data_splits", "2021_to_2023_split")
    os.makedirs(save_dir, exist_ok=True)
    test_save_path = os.path.join(save_dir, "test_data.pt") 

    test_data_list = load_graphml_files(years)
    test_data_batch = Batch.from_data_list(test_data_list)
    test_data_batch = normalize_test_features(test_data_batch)
    print("\nTest Data Statistics:")
    print_batch_shape(test_data_batch)

    torch.save(test_data_batch, test_save_path)
    print(f"\nTest data saved to: {test_save_path}")


if __name__ == "__main__":
    main()
