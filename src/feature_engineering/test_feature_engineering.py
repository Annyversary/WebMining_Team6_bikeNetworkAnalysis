"""
Test script to verify that the network feature engineering script works correctly.
This script tests adding network features to a single graphml file and verifies the results.
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

# Add the parent directory to the path to import the add_network_features module
sys.path.append(str(Path(__file__).parent.parent))
from feature_engineering.add_network_features import add_network_features, euclidean_distance_meters

def test_feature_engineering(input_file, output_dir=None):
    """
    Test adding network features to a single graphml file.
    
    Parameters:
    -----------
    input_file : str
        Path to the input graphml file
    output_dir : str, optional
        Directory to save the enhanced graphml file and verification plots
    """
    print(f"Testing feature engineering on {input_file}")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_filename}_with_features.graphml")
        plots_dir = os.path.join(output_dir, "verification_plots")
        os.makedirs(plots_dir, exist_ok=True)
    else:
        output_file = None
        plots_dir = None
    
    # Read the graph
    try:
        print("Reading the graph...")
        G = nx.read_graphml(input_file)
        original_nodes = G.number_of_nodes()
        original_edges = G.number_of_edges()
        print(f"Graph loaded with {original_nodes} nodes and {original_edges} edges")
        
        # Get a few sample node IDs for later verification
        sample_nodes = list(G.nodes())[:5] if G.nodes() else []
        
        # Add network features
        print("Adding network features...")
        G = add_network_features(G)
        
        # Verify that nodes and edges weren't altered
        assert G.number_of_nodes() == original_nodes, "Number of nodes changed during feature engineering"
        assert G.number_of_edges() == original_edges, "Number of edges changed during feature engineering"
        
        # Verify that features were added to nodes
        node_features_to_check = ['in_degree', 'out_degree', 'closeness_centrality', 'k_core', 'betweenness_centrality']
        for node in sample_nodes:
            for feature in node_features_to_check:
                assert feature in G.nodes[node], f"Feature '{feature}' not added to node {node}"
        print("Node features verified")
        
        # Verify that distance is added to edges
        edge_features_to_check = ['distance_m']
        for u, v in list(G.edges())[:5]:
            for feature in edge_features_to_check:
                assert feature in G[u][v], f"Feature '{feature}' not added to edge ({u}, {v})"
                # Verify that distance is positive
                assert float(G[u][v]['distance_m']) >= 0, f"Distance is negative for edge ({u}, {v})"
        print("Edge features verified")
        
        # Verify in-degree and out-degree
        for node in sample_nodes:
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            node_data = G.nodes[node]
            assert int(node_data['in_degree']) == in_degree, f"In-degree mismatch for node {node}"
            assert int(node_data['out_degree']) == out_degree, f"Out-degree mismatch for node {node}"
        print("In-degree and out-degree verified")
        
        # Create plots for verification if output_dir is specified
        if plots_dir:
            print("Creating verification plots...")
            
            # Plot feature distributions
            for feature in node_features_to_check:
                values = [float(G.nodes[node][feature]) for node in G.nodes()]
                plt.figure(figsize=(8, 5))
                plt.hist(values, bins=20, alpha=0.7)
                plt.title(f'Distribution of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(plots_dir, f"{feature}_distribution.png"))
                plt.close()
            
            # Plot distance distribution
            distances = []
            for u, v, data in G.edges(data=True):
                if 'distance_m' in data:
                    distances.append(float(data['distance_m']))
            
            plt.figure(figsize=(8, 5))
            plt.hist(distances, bins=20, alpha=0.7)
            plt.title('Distribution of Edge Distances')
            plt.xlabel('Distance (m)')
            plt.ylabel('Frequency')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plots_dir, "distance_distribution.png"))
            plt.close()
            
            print(f"Verification plots saved to {plots_dir}")
        
        # Save the enhanced graph if output file is specified
        if output_file:
            print(f"Saving enhanced graph to {output_file}")
            nx.write_graphml(G, output_file)
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the network feature engineering script.")
    parser.add_argument("--input", required=True, help="Input graphml file to test")
    parser.add_argument("--output", help="Output directory for the enhanced graphml file and verification plots")
    args = parser.parse_args()
    
    success = test_feature_engineering(args.input, args.output)
    sys.exit(0 if success else 1)