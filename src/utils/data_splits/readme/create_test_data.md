## Create Test Set

### Importing Dependencies

We import the necessary libraries and functions, ensuring that all required modules and helper functions are properly integrated.

### Executing the Pipeline for Creating the Test Dataset

This script defines a `main` function that orchestrates the complete pipeline for generating a test dataset for Graph Neural Networks (GNNs). The previously defined functions are called sequentially to:

1. **Load the graph data**: The GraphML files for the specified years are loaded into PyTorch Geometric `Data` objects.
2. **Batch the graphs**: The individual graphs are then combined into a single batched dataset.
3. **Feature normalization**: The features are normalized using the scalers that were previously initialized during training.
4. **Print batch statistics**: To gain insight into the structure of the test set, the number of graphs, nodes, and edges is printed.

The resulting test dataset is then saved for later use in model evaluation.
