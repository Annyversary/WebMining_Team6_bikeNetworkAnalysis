## Wrapper Method: `transform_networkx_into_pyg`

This wrapper provides a method for converting a **NetworkX graph** into a **PyTorch Geometric `Data` object**, including all available **node and edge attributes**. This step is essential because **PyTorch Geometric does not natively support `.graphml` files** and therefore cannot load them directly. The conversion enables further processing and training with Graph Neural Networks (GNNs) in PyTorch Geometric.


### Purpose and Considerations

NetworkX is a widely used library for creating and manipulating graphs in Python. However, to use these graphs within a PyTorch Geometric pipeline, they must first be converted into the framework’s specific `Data` format. This object typically includes:

- `x`: Node features (Geographical coordinates such as longitude and latitude) 
*(*Additional nodes after feature engineering*)*
- `edge_index`: Graph connectivity in COO format
- `edge_attr`: Edge attributes

This wrapper ensures that all relevant node and edge information is preserved during the conversion.