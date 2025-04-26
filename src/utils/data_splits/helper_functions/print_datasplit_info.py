def print_batch_shape(batch):
    """
    Prints detailed information about the structure of a PyTorch Geometric batch.

    Parameters:
    -----------
    batch : torch_geometric.data.Batch
        A batched PyTorch Geometric Data object containing multiple graphs.

    Outputs:
    --------
    Prints the number of graphs, total number of nodes and edges,
    and the shapes of node features, edge indices, and edge attributes (if present).
    """

    num_graphs = batch.num_graphs
    print(f"Number of graphs in batch: {num_graphs}")
    
    # Nodes
    num_nodes = batch.x.shape[0]
    print(f"Number of nodes: {num_nodes}")
    print(f"Node feature shape: {batch.x.shape}")
    
    # Edges
    num_edges = batch.edge_index.shape[1]  
    print(f"Number of edges: {num_edges}")
    print(f"Edge index shape: {batch.edge_index.shape}")
    
    if batch.edge_attr is not None:
        print(f"Edge attributes shape: {batch.edge_attr.shape}")
    
    if batch.x is not None:
        print(f"Node features shape: {batch.x.shape}")
    else:
        print("No node features available.")
