"""
Script to add network features to nodes in graphml files.
Features added:
- In-degree
- Out-degree
- Closeness centrality
- K-core values
- Betweenness centrality (currently disabled for performance reasons)
- Distance between nodes (edge feature, in meters)
"""

import os
import glob
import networkx as nx
from tqdm import tqdm
import argparse
import math

def euclidean_distance_meters(lon1, lat1, lon2, lat2):
    """
    Calculate the Euclidean distance between two points
    on earth (specified in decimal degrees) and convert to meters
    
    For short distances, this approximation is sufficient and avoids
    calculating Earth's curvature.
    
    Parameters:
    -----------
    lon1, lat1 : float
        Longitude and latitude of point 1
    lon2, lat2 : float
        Longitude and latitude of point 2
        
    Returns:
    --------
    float
        Distance in meters
    """
    # Approximate conversion to meters (for short distances)
    # 1 degree of latitude is approximately 111,000 meters
    # 1 degree of longitude varies with latitude, but is roughly 111,000 * cos(lat) meters
    
    # Average latitude for longitude scaling
    avg_lat_radians = math.radians((lat1 + lat2) / 2)
    
    # Calculate distances in meters
    y_distance = (lat2 - lat1) * 111000
    x_distance = (lon2 - lon1) * 111000 * math.cos(avg_lat_radians)
    
    # Euclidean distance
    distance_meters = math.sqrt(x_distance**2 + y_distance**2)
    
    return distance_meters

def add_network_features(graph):
    """
    Add network features to the nodes in the graph.
    
    Parameters:
    -----------
    graph : networkx.DiGraph
        The directed graph to add features to
    
    Returns:
    --------
    networkx.DiGraph
        Graph with node features added
    """
    print("Computing in-degree and out-degree...")
    # Add in-degree and out-degree
    for node in graph.nodes():
        graph.nodes[node]['in_degree'] = graph.in_degree(node)
        graph.nodes[node]['out_degree'] = graph.out_degree(node)
    
    print("Computing distances between nodes...")
    # Add distance between nodes as an edge feature
    for u, v in graph.edges():
        # Extract longitude and latitude from node IDs or attributes
        if 'lon' in graph.nodes[u] and 'lat' in graph.nodes[u] and 'lon' in graph.nodes[v] and 'lat' in graph.nodes[v]:
            # If coordinates are stored in node attributes
            lon1 = float(graph.nodes[u]['lon'])
            lat1 = float(graph.nodes[u]['lat'])
            lon2 = float(graph.nodes[v]['lon'])
            lat2 = float(graph.nodes[v]['lat'])
        else:
            # If coordinates are part of the node ID - parse from string like "(9.717, 52.373)"
            try:
                # Remove parentheses and split by comma
                u_coords = u.strip('()').split(',')
                v_coords = v.strip('()').split(',')
                
                lon1 = float(u_coords[0])
                lat1 = float(u_coords[1])
                lon2 = float(v_coords[0])
                lat2 = float(v_coords[1])
            except (ValueError, IndexError):
                print(f"Warning: Could not parse coordinates from node IDs {u}, {v}. Skipping distance calculation.")
                continue
                
        # Calculate distance in meters
        distance = euclidean_distance_meters(lon1, lat1, lon2, lat2)
        # Round to 3 decimal places
        rounded_distance = round(distance, 3)
        graph[u][v]['distance_m'] = rounded_distance
    
    print("Computing closeness centrality...")
    # Add closeness centrality
    closeness = nx.closeness_centrality(graph)
    # Round closeness centrality values to 3 decimal places
    closeness = {node: round(value, 3) for node, value in closeness.items()}
    nx.set_node_attributes(graph, closeness, 'closeness_centrality')
    
    print("Computing k-core values...")
    # Add k-core value directly on the directed graph
    # This preserves the directional nature of the bicycle network
    k_core = nx.core_number(graph)
    nx.set_node_attributes(graph, k_core, 'k_core')
    
    # print("Computing betweenness centrality...") This has been disabled for performance reasons
    # # Add betweenness centrality (this can be slow for large graphs)
    # betweenness = nx.betweenness_centrality(graph)
    # # Round betweenness centrality values to 3 decimal places
    # betweenness = {node: round(value, 3) for node, value in betweenness.items()}
    # nx.set_node_attributes(graph, betweenness, 'betweenness_centrality')
    
    return graph

def process_file(input_file, output_file=None):
    """
    Process a single graphml file, add features, and save to output file.
    
    Parameters:
    -----------
    input_file : str
        Path to input graphml file
    output_file : str, optional
        Path to output graphml file. If None, will overwrite input file.
    """
    if output_file is None:
        output_file = input_file
        
    print(f"Processing {input_file}...")
    try:
        # Read the graph
        G = nx.read_graphml(input_file)
        print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Add features
        G = add_network_features(G)
        
        # Save the graph
        nx.write_graphml(G, output_file)
        print(f"Graph saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def process_directory(input_dir, output_dir=None):
    """
    Process all graphml files in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Path to directory containing graphml files
    output_dir : str, optional
        Path to directory to save output graphml files. If None, will overwrite input files.
    """
    # Find all graphml files in the directory
    graphml_files = glob.glob(os.path.join(input_dir, "*.graphml"))
    
    if not graphml_files:
        print(f"No graphml files found in {input_dir}")
        return
    
    print(f"Found {len(graphml_files)} graphml files in {input_dir}")
    
    for input_file in tqdm(graphml_files):
        if output_dir is not None:
            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output file path
            filename = os.path.basename(input_file)
            output_file = os.path.join(output_dir, filename)
        else:
            output_file = input_file
        print(f"Processing {input_file}...")
        process_file(input_file, output_file)

def main():
    parser = argparse.ArgumentParser(description="Add network features to graphml files.")
    parser.add_argument("--input", required=True, help="Input graphml file or directory")
    parser.add_argument("--output", help="Output graphml file or directory (optional, defaults to overwriting input)")
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        process_directory(args.input, args.output)
    else:
        process_file(args.input, args.output)

if __name__ == "__main__":
    main()