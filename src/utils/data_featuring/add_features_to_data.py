#!/usr/bin/env python
"""
Utility script to add network features to all GraphML files in data/graphml.
This script iterates through the yearly folders (2021, 2022, 2023, 2024) and processes
all GraphML files in each folder using add_network_features.py. The featured versions
are saved to data/data_featured, maintaining the same folder structure.

Usage:
    python src/utils/data_featuring/add_features_to_data.py

"""

import os
import sys

# Add the current project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# Now import from feature_engineering
from src.feature_engineering.add_network_features import process_directory


def ensure_directory_exists(directory):
    """
    Create a directory if it doesn't exist.
    
    Parameters:
    -----------
    directory : str
        Path to the directory to create
    """
    os.makedirs(directory, exist_ok=True)
    print(f"Directory ensured: {directory}")


def process_all_graphml_files():
    """
    Process all GraphML files in data/graphml and save featured versions to data/data_featured.
    """
    # Define paths
    graphml_base_dir = os.path.join("data", "graphml")
    featured_base_dir = os.path.join("data", "data_featured")
    
    # Ensure the base output directory exists
    ensure_directory_exists(featured_base_dir)
    
    # Process each year directory (2021, 2022, 2023, 2024)
    years = ['2021', '2022', '2023', '2024']
    
    for year in years:
        input_dir = os.path.join(graphml_base_dir, year)
        output_dir = os.path.join(featured_base_dir, year)
        
        # Check if the input directory exists
        if os.path.exists(input_dir):
            # Ensure the output directory exists
            ensure_directory_exists(output_dir)
            
            print(f"\nProcessing files for year {year}...")
            # Use the process_directory function from add_network_features.py
            process_directory(input_dir, output_dir)
        else:
            print(f"Warning: Input directory not found: {input_dir}")
    
    print("\nAll GraphML files have been processed and featured versions saved.")


if __name__ == "__main__":
    print("Starting to add network features to all GraphML files...")
    process_all_graphml_files()
    print("Done!")