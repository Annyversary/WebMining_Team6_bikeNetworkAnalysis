"""
Script to visualize network features in graphml files.
Creates histograms and scatter plots of node features and edge features.
"""

import os
import glob
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm

def plot_feature_histogram(feature_values, feature_name, output_file=None):
    """
    Plot histogram of feature values.
    
    Parameters:
    -----------
    feature_values : list
        List of feature values
    feature_name : str
        Name of the feature
    output_file : str, optional
        Path to save the plot. If None, will display the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(feature_values, bins=30, alpha=0.7)
    plt.title(f'Distribution of {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved histogram to {output_file}")
    else:
        plt.show()

def plot_feature_scatter(x_values, y_values, x_name, y_name, output_file=None):
    """
    Plot scatter plot of two features.
    
    Parameters:
    -----------
    x_values : list
        List of values for x-axis
    y_values : list
        List of values for y-axis
    x_name : str
        Name of the x feature
    y_name : str
        Name of the y feature
    output_file : str, optional
        Path to save the plot. If None, will display the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, alpha=0.5)
    plt.title(f'Relationship between {x_name} and {y_name}')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid(alpha=0.3)
    
    # Add a best fit line
    if len(x_values) > 1:
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        plt.plot(np.array(sorted(x_values)), p(np.array(sorted(x_values))), "r--", alpha=0.8)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved scatter plot to {output_file}")
    else:
        plt.show()

def plot_edge_feature_histogram(feature_values, feature_name, output_file=None):
    """
    Plot histogram of edge feature values.
    
    Parameters:
    -----------
    feature_values : list
        List of feature values
    feature_name : str
        Name of the feature
    output_file : str, optional
        Path to save the plot. If None, will display the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(feature_values, bins=30, alpha=0.7)
    plt.title(f'Distribution of {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved histogram to {output_file}")
    else:
        plt.show()

def visualize_graph_features(input_file, output_dir=None):
    """
    Visualize features of a single graphml file.
    
    Parameters:
    -----------
    input_file : str
        Path to graphml file
    output_dir : str, optional
        Directory to save plots. If None, will display plots instead of saving.
    """
    try:
        print(f"Loading graph from {input_file}...")
        G = nx.read_graphml(input_file)
        print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.splitext(os.path.basename(input_file))[0]
        
        # Extract node feature values
        node_features = {
            'in_degree': [],
            'out_degree': [],
            'closeness_centrality': [],
            'k_core': [],
            'betweenness_centrality': []
        }
        
        for node in G.nodes():
            node_data = G.nodes[node]
            for feature_name in node_features:
                if feature_name in node_data:
                    # Convert to float for consistent plotting
                    node_features[feature_name].append(float(node_data[feature_name]))
        
        # Extract edge feature values
        edge_features = {
            'distance_m': []  # Updated to look for distance_m instead of distance_km
        }
        
        for u, v, key, data in G.edges(data=True, keys=True):
            for feature_name in edge_features:
                if feature_name in data:
                    edge_features[feature_name].append(float(data[feature_name]))
        
        # Check which node features are available in the graph
        available_node_features = [f for f in node_features if len(node_features[f]) > 0]
        
        # Check which edge features are available in the graph
        available_edge_features = [f for f in edge_features if len(edge_features[f]) > 0]
        
        if not available_node_features and not available_edge_features:
            print("No network features found in the graph. Run add_network_features.py first.")
            return
        
        print(f"Found node features: {available_node_features}")
        print(f"Found edge features: {available_edge_features}")
        
        # Plot histograms for each node feature
        for feature_name in available_node_features:
            if output_dir:
                output_file = os.path.join(output_dir, f"{base_filename}_{feature_name}_hist.png")
            else:
                output_file = None
                
            plot_feature_histogram(node_features[feature_name], feature_name, output_file)
        
        # Plot histograms for each edge feature
        for feature_name in available_edge_features:
            if output_dir:
                output_file = os.path.join(output_dir, f"{base_filename}_{feature_name}_edge_hist.png")
            else:
                output_file = None
                
            plot_edge_feature_histogram(edge_features[feature_name], feature_name, output_file)
        
        # Plot scatter plots for each pair of node features
        for i, feature1 in enumerate(available_node_features):
            for feature2 in available_node_features[i+1:]:
                if output_dir:
                    output_file = os.path.join(output_dir, 
                                               f"{base_filename}_{feature1}_vs_{feature2}_scatter.png")
                else:
                    output_file = None
                    
                plot_feature_scatter(node_features[feature1], node_features[feature2], 
                                     feature1, feature2, output_file)
        
        # Add some relation plots between node and edge features if both exist
        if available_node_features and 'distance_m' in available_edge_features:  # Updated to check for distance_m
            # Get the average distance for each node (considering outgoing edges)
            node_avg_distances = {}
            for u, v, key, data in G.edges(data=True, keys=True):
                if 'distance_m' in data:  # Updated to look for distance_m
                    if u not in node_avg_distances:
                        node_avg_distances[u] = []
                    node_avg_distances[u].append(float(data['distance_m']))  # Updated to use distance_m
            
            # Calculate average
            node_avg_distance_values = {}
            for node, distances in node_avg_distances.items():
                if distances:
                    node_avg_distance_values[node] = sum(distances) / len(distances)
            
            # Create plots relating node features to average distance
            for feature_name in available_node_features:
                # Collect data points
                x_values = []
                y_values = []
                for node in G.nodes():
                    if node in node_avg_distance_values and feature_name in G.nodes[node]:
                        x_values.append(float(G.nodes[node][feature_name]))
                        y_values.append(node_avg_distance_values[node])
                
                if x_values and y_values:
                    if output_dir:
                        output_file = os.path.join(output_dir, 
                                                 f"{base_filename}_{feature_name}_vs_avg_distance_scatter.png")
                    else:
                        output_file = None
                        
                    plot_feature_scatter(x_values, y_values, feature_name, 'Average Outgoing Distance (m)',  # Updated label
                                        output_file)
    
    except Exception as e:
        print(f"Error visualizing {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Visualize network features in graphml files.")
    parser.add_argument("--input", required=True, help="Input graphml file or directory")
    parser.add_argument("--output", help="Output directory for plots (optional, defaults to displaying plots)")
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        # Process all graphml files in the directory
        graphml_files = glob.glob(os.path.join(args.input, "*.graphml"))
        
        if not graphml_files:
            print(f"No graphml files found in {args.input}")
            return
        
        print(f"Found {len(graphml_files)} graphml files in {args.input}")
        
        for input_file in tqdm(graphml_files):
            visualize_graph_features(input_file, args.output)
    else:
        # Process a single file
        visualize_graph_features(args.input, args.output)

if __name__ == "__main__":
    main()