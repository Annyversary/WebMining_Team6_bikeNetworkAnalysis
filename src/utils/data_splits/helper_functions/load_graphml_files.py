import networkx as nx
import os

from wrapper.transform_networkx_into_pyg import transform_networkx_into_pyg

def load_graphml_files(years=[2021, 2022, 2023]):
    """
    Loads multiple directed graph files in GraphML format and converts them 
    into PyTorch Geometric (PyG) Data objects.

    Parameters:
    -----------
    years : list of int (default=[2021, 2022, 2023])
        List of years for which graph files should be loaded. 
        Assumes 12 monthly files per year.

    Returns:
    --------
    data_list : list of torch_geometric.data.Data
        List of PyG data objects created from the loaded NetworkX graphs.
    """

    data_list = []

    for year in years:
        for i in range(12):
            graph_path = os.path.join('data', 'graphml', str(year), f"bike_network_{year}_{i}.graphml")
            graph_path = os.path.abspath(graph_path)

            if not os.path.exists(graph_path):
                print(f"[WARN] File not found: {graph_path}")
                continue

            G_nx = nx.read_graphml(graph_path)
            G_nx = nx.DiGraph(G_nx)
            data = transform_networkx_into_pyg(G_nx)
            data_list.append(data)

    print(f"Number of loaded graphs: {len(data_list)}")
    return data_list