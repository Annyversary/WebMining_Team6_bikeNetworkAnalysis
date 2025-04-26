## Note on the GML File and Network Modeling: `build_network_from_gml`

Our quarterly GML datasets are not GML files in the traditional sense of the **Graph Modelling Language** as expected by `networkx.read_gml()`. Instead, they use the **Geographic Markup Language (GML)** – an XML-based format for representing geographical features such as lines, points, and polygons.

While `networkx.read_gml()` expects a GML file structured as a graph with named nodes and edges, the dataset in question contains georeferenced lines in the form of `LineString` geometries. These lines represent bicycle paths defined by their coordinates but do not explicitly define a network with uniquely labeled nodes and directed edges.


To build a `networkx`-compatible network from this file, the following steps were required:

1. **Load the file using `geopandas`**: The geographic GML format can be read directly with GeoPandas.
2. **Extract geometries**: The geometries in the file represent paths, and each geometry can be used to define edges in the network.
3. **Determine the start and end points** of each line: These serve as nodes in the graph.
4. **Create a directed graph with `networkx`**: Each line (edge) can be enriched with attributes like `TRACKS_FWD`, `TRACKS_BAC`, `SPEED_REL`, `YEAR`, and `MONTH`. The forward (`TRACKS_FWD`) and backward (`TRACKS_BAC`) edges are stored separately, allowing for directional analysis of the paths.

These steps make it possible to transform geospatial data into a network structure suitable for advanced analysis such as **centrality measures**, **clustering**, or **prediction tasks**.

Due to rounding to three decimal places in a previous processing step, the following issue can occur:
An edge `(start, end)` may already exist in the graph but lacks the `'tracks_fwd'` attribute because it was originally stored in the reverse direction and instead has the `'tracks_bac'` attribute. We have accounted for this issue in our implementation logic. 

If this problem is still unclear, the following visualization illustrates the issue in more detail:

![Visualization](../../../images/Aggregating_TRACKS.png)


## Generating Graphs for Each Year and Quarter: `generate_networks()`

The function `generate_networks()` automates the process of generating bike network graphs for each month in the years 2021 through 2024. The function iterates over all quarters and months within each year and processes the corresponding GML files for each quarter.
