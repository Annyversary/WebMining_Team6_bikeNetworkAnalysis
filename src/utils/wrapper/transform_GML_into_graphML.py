import geopandas as gpd
import networkx as nx
import os

def build_network_from_gml(file_path, month, year):
    """
    Creates a directed bike network graph from the provided GML file, 
    filtered by the specified month and year. The graph is then saved 
    as a GraphML file.

    Parameters:
    file_path (str): The file path to the GML file containing the bike network data.
    month (int): The month to filter the data.
    year (int): The year to filter the data.
    """
    
    # 1. Load the GML file
    gdf = gpd.read_file(file_path)

    # 2. Initialize the network (directed graph)
    G = nx.DiGraph()

    # 3. Iterate over each row in the GeoDataFrame
    for idx, row in gdf.iterrows():
        # Only process rows with the specified month and year
        if row.get('MONTH') != month or row.get('YEAR') != year:
            continue

        # Get the geometry of the row
        geom = row.geometry

        # Process each line in the geometry (in case of multiple geometries in a row)
        for line in geom.geoms:
            coords = list(line.coords)
            # Round the coordinates to 3 decimal places
            coords_rounded = [(round(x, 3), round(y, 3)) for x, y in coords]

            start = tuple(coords_rounded[0])
            end = tuple(coords_rounded[-1])

            # Skip if the start and end points are the same (no edge)
            if start == end:
                continue

            # Add the start and end nodes if they don't already exist in the graph
            if not G.has_node(start):
                G.add_node(start, lon=start[0], lat=start[1])
            if not G.has_node(end):
                G.add_node(end, lon=end[0], lat=end[1])


            # Add or update the forward edge (tracks_fwd)
            # Due to rounding to 3 decimal places earlier, the following scenario arises:
            # The edge (start, end) may already exist but does not have the 'tracks_fwd' feature, as it was previously stored as a reverse edge 
            # and instead possesses the tracks_bac feature. Here, we ensure the correct addition of the tracks.
            if G.has_edge(start, end) and 'tracks_fwd' in G[start][end]:
                G[start][end]['tracks_fwd'] += row.get('TRACKS_FWD', 0)
            elif G.has_edge(start, end) and 'tracks_bac' in G[start][end]: 
                G[start][end]['tracks_bac'] += row.get('TRACKS_BAC', 0) 
            else:
                # Create the forward edge if it doesn't exist
                G.add_edge(
                    start, end,
                    id=row.get('ID'),
                    tracks_fwd=row.get('TRACKS_FWD'),
                    year=row.get('YEAR'),
                    month=row.get('MONTH'),
                    speed_rel=row.get('SPEED_REL')
                )

            # Add or update the backward edge (tracks_bac)
            # At this point, the same thing happens as above, but in reverse, to maintain the correct relationship 
            # between forward and backward tracks during the addition.
            if G.has_edge(end, start) and 'tracks_bac' in G[end][start]:
                G[end][start]['tracks_bac'] += row.get('TRACKS_FWD', 0)
            elif G.has_edge(end, start) and 'tracks_fwd' in G[end][start]:
                G[end][start]['tracks_fwd'] += row.get('TRACKS_BAC', 0)
            else:
                # Create the backward edge if it doesn't exist
                G.add_edge(
                    end, start,
                    id=row.get('ID'),
                    tracks_bac=row.get('TRACKS_BAC'),
                    year=row.get('YEAR'),
                    month=row.get('MONTH'),
                    speed_rel=row.get('SPEED_REL')
                )
     
    # 4. Iterate over all edges in the graph and rename the "track" attribute, as PyTorch Geometric expects all edges to have the same attributes when converted into a Data object.
    for u, v, attr in G.edges(data=True):
        if 'tracks_fwd' in attr:
            attr['tracks'] = attr.pop('tracks_fwd')  # Rename 'tracks_fwd' to 'tracks'
        if 'tracks_bac' in attr:
            attr['tracks'] = attr.pop('tracks_bac')  # Rename 'tracks_bac' to 'tracks'

    # 5. Print the number of nodes and edges in the graph
    print(f"Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
    print("First 5 nodes:", list(G.nodes)[:5])
    print("First 5 edges:", list(G.edges(data=True))[:5])

    # 6. Save the network as a GraphML file
    output_dir = os.path.join('data', 'graphml', str(year))
    os.makedirs(output_dir, exist_ok=True) 
    output_path = os.path.join(output_dir, f"bike_network_{year}_{month}.graphml")

    # 7. Save the graph in GraphML format
    nx.write_graphml(G, output_path)
    print(f"Graph saved at: {output_path}")



def generate_networks():
    """
    Generates directed bike network graphs for each month between 2021 and 2024.

    For each month, the function:
    - Searches for a GML file based on a predefined directory structure: 
      ../../../data/gml/{year}/bike_citizens_rh_{quarter}_{year}.gml
    - Filters the data by the corresponding month and year.
    - Calls `build_network_from_gml` to create and save the corresponding GraphML file.

    Notes:
    - Expects GML files organized by year and quarter.
    - If a GML file does not exist, a message is printed.
    - Saves output GraphML files into ../../../data/graphml/{year}/.
    """

    # Years from 2021 to 2024
    years = range(2021, 2025)
    
    # Quarters and months
    quarters = {
        "Q1": range(0, 3),   # January to March
        "Q2": range(3, 6),   # April to June
        "Q3": range(6, 9),   # July to September
        "Q4": range(9, 12)   # October to December
    }
    
    # Loop through each year and each quarter, then each month
    for year in years:
        for quarter, months in quarters.items():
            for month in months:
                # Create the file path based on the schema
                file_path = os.path.join("data", "gml", str(year), f"bike_citizens_rh_{quarter}_{year}.gml")

                # Check if the file exists
                if os.path.exists(file_path):
                    print(f"Processing file: {file_path} (Year: {year}, Quarter: {quarter}, Month: {month})")
                    build_network_from_gml(file_path, month, year)
                else:
                    print(f"File not found: {file_path} (Year: {year}, Quarter: {quarter}, Month: {month})")


if __name__ == "__main__":
    generate_networks()
