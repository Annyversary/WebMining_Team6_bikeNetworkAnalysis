"""
A script to detect outliers in the 'tracks' attribute of edges in bike network GraphML files.
For each month of each year, it calculates the average 'tracks' value and identifies the top 10 edges with the highest 'tracks' values.
"""

import os
import glob
import networkx as nx
from collections import defaultdict
import numpy as np

def get_month_name(month_num):
    """Convert month number (0-11) to month name."""
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    return month_names[month_num]

def analyze_tracks_outliers():
    """
    Analyze GraphML files for outliers in the 'tracks' attribute.
    For each month of each year, calculate the average 'tracks' value and
    identify the top 10 edges with the highest 'tracks' values.
    """
    # Dictionary to store edges by year and month
    edges_by_year_month = defaultdict(lambda: defaultdict(list))
    
    # Path to the data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    
    # Check if data_featured directory exists and use it if available (it has more features)
    if os.path.exists(os.path.join(data_dir, 'data_featured')):
        graphml_dir = os.path.join(data_dir, 'data_featured')
        print("Using data_featured directory for analysis.")
    else:
        graphml_dir = os.path.join(data_dir, 'graphml')
        print("Using graphml directory for analysis.")
    
    # Get all GraphML files
    graphml_files = []
    for year_dir in glob.glob(os.path.join(graphml_dir, '*')):
        year = os.path.basename(year_dir)
        for graphml_file in glob.glob(os.path.join(year_dir, '*.graphml')):
            graphml_files.append((year, graphml_file))
    
    print(f"Found {len(graphml_files)} GraphML files to analyze.")
    
    # Process each GraphML file
    for year, file_path in sorted(graphml_files):
        try:
            G = nx.read_graphml(file_path)
            
            # Extract edges with 'tracks' attribute
            for u, v, data in G.edges(data=True):
                if 'tracks' in data:
                    # Get month from file or from edge attribute
                    if 'month' in data:
                        month = int(data['month'])
                    else:
                        # Extract month from filename (e.g., bike_network_2021_0.graphml where 0 is January)
                        filename = os.path.basename(file_path)
                        month = int(filename.split('_')[-1].split('.')[0])
                    
                    # Store edge info (nodes, tracks value, and source file)
                    edge_info = {
                        'source': u,
                        'target': v,
                        'tracks': int(data['tracks']),
                        'file': file_path
                    }
                    edges_by_year_month[year][month].append(edge_info)
            
            print(f"Processed {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Analyze edges by year and month
    for year in sorted(edges_by_year_month.keys()):
        for month, edges in sorted(edges_by_year_month[year].items()):
            if not edges:
                continue
                
            month_name = get_month_name(month)
            tracks_values = [edge['tracks'] for edge in edges]
            avg_tracks = np.mean(tracks_values)
            
            print(f"\n{'='*80}")
            print(f"Year: {year}, Month: {month_name} (Month number: {month})")
            print(f"Total edges with 'tracks' attribute: {len(edges)}")
            print(f"Average 'tracks' value: {avg_tracks:.2f}")
            print(f"{'='*80}")
            
            # Get top 10 edges with highest 'tracks' values
            top_edges = sorted(edges, key=lambda x: x['tracks'], reverse=True)[:10]
            
            print("\nTop 10 edges with highest 'tracks' values:")
            print(f"{'Source Node':^20} | {'Target Node':^20} | {'Tracks':^10} | {'File'}")
            print(f"{'-'*20} | {'-'*20} | {'-'*10} | {'-'*40}")
            
            for i, edge in enumerate(top_edges, 1):
                print(f"{str(edge['source']):^20} | {str(edge['target']):^20} | {edge['tracks']:^10} | {os.path.basename(edge['file'])}")

if __name__ == "__main__":
    analyze_tracks_outliers()