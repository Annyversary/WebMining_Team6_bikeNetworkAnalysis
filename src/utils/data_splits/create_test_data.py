import os
import torch

from helper_functions.load_graphml_files import load_graphml_files
from normalize_data.normalize_test_data import normalize_test_features

def main(years=[2024]):
    save_dir = os.path.join("data", "data_splits", "test_data_monthly")
    os.makedirs(save_dir, exist_ok=True)

    test_data_list = load_graphml_files(years)

    # Optional: normalize each Data object individually
    for i, data in enumerate(test_data_list):
        data = normalize_test_features(data)
        print(f"\nStatistics for month {i+1:02d}:")

        # Save each file individually
        month_save_path = os.path.join(save_dir, f"month_{i+1:02d}.pt")
        torch.save(data, month_save_path)
        print(f"Saved to: {month_save_path}")

if __name__ == "__main__":
    main()
