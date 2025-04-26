### Inspecting the Structure of a Graph Batch

To better understand the structure of the batched graph data used in our pipeline, we implemented a utility function called `print_batch_shape`. This function takes a PyTorch Geometric `Batch` object as input and prints key statistics about its contents.

Specifically, it outputs:
- The number of individual graphs in the batch
- The total number of nodes and edges
- The shape of the node feature matrix (`x`)
- The shape of the edge index tensor (`edge_index`)
- The shape of the edge attributes tensor (`edge_attr`), if available

This function is useful for debugging and verifying the correct assembly of graph batches prior to model training or evaluation.
