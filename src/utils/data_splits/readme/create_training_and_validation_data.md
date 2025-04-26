## Create Training and Validation Set

### Importing Dependencies

We import the necessary libraries and functions, ensuring that all required modules and helper functions are properly integrated.

### Train-Validation Split

For predicting edge attributes (e.g., `tracks`), an 80/20 train/validation split is applied to the **existing edges within each graph**. This means that the edges are randomly split into training and validation sets, while the nodes remain unchanged during the process.

In our application, the **nodes represent physically existing bike stations**, which typically do not change or only change very infrequently. The aim of the analysis is to model the **connections between stations**, i.e., to understand and predict how many bicycles move along certain routes (in other words: edges with weights).

A **node-level split** (i.e., an 80/20 split of the nodes themselves) would mean that some stations would be completely unseen during training. This would not be meaningful because:

- The **stations themselves are not the prediction target**;
- It is the **relationships or transitions between the stations (edges)** that should be modeled;
- In deployment, **all stations are known** (they are physically installed in the system);

Initially, we wanted to use the `RandomLinkSplit()` function, but this is designed for classic link prediction – i.e., binary classification. It adds both positive examples (existing edges) AND negative examples (non-existing edges). Since our task is an edge attribute regression task, this method is unsuitable, and we manually implemented the split mechanism using a random permutation of the edges. The `edge_attr_key_index` parameter is used to specify which edge attribute (e.g., `tracks`) is the target variable for prediction.

### Executing the Pipeline for Creating Training and Validation Data Sets

This script defines a `main` function that orchestrates the entire pipeline for generating training and validation splits for Graph Neural Networks (GNNs). The previously defined functions are called sequentially to:

1. **Load the graph data**: The GraphML files for the specified years are loaded into PyTorch Geometric `Data` objects.
2. **Split the edges** into training and validation sets: The edges of each graph are randomly split into training and validation subsets, with the specified ratio for validation data (default 20%).
3. **Feature normalization**: The features of the training and validation sets are normalized.
4. **Print batch statistics**: To gain insight into the structure of the train and validation sets, the number of graphs, nodes, and edges is printed.

The training and validation data are then saved for later use in model training.
