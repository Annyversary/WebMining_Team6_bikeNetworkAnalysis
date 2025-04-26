### Loading and Preparing Graph Data from GraphML Files: `load_graphml_files`

The `load_graphml_files` function loads a series of bicycle traffic network graphs stored in GraphML format and prepares them for training with PyTorch Geometric (PyG). The objective is to convert each monthly graph into a format compatible with Graph Neural Networks (GNNs), ensuring that both node and edge features are retained.

Each NetworkX graph is converted into a PyG `Data` object using a custom helper function `transform_networkx_into_pyg`. This function ensures that essential node and edge attributes such as:

- **Node attributes**:
  - `lon` (longitude),
  - `lat` (latitude),
  
- **Edge attributes**:
  - `speed_rel` (relative speed),
  - `month` (for cyclical encoding of months),
  - `year` (the year of the traffic data),
  - `id` (an identifier for a certain trackfrom A to B),
  - `tracks` (the number of bicycles traveling from the starting to the ending point),

are preserved during the conversion process.

PyG expects data in a specific structure, particularly when both node and edge attributes are used in models like GATv2.

`data_list` contains multiple `torch_geometric.data.Data` objects, each representing a graph.
