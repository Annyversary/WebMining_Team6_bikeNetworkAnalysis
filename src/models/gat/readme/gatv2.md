## Graph Attention Network (GAT)

### Importing Dependencies

We import the necessary libraries and functions, ensuring that all required modules and helper functions are properly integrated.

## Implementing a GAT Model

### From GAT to GATv2

We initially started by implementing a vanilla GAT model—an advanced type of Graph Neural Network (GNN) that leverages **attention mechanisms**. However, we soon realized that this model is not capable of learning from **edge attributes**, which is essential for our task. This limitation became especially critical because our original dataset does not contain any **node attributes** at all.

This led us to adopt the **GATv2** model, which is specifically designed to aggregate node features while also considering **edge attributes** during message passing. It is more suitable for our purposes.

The GATv2 model expects the following **inputs**:
- `x`: Node features → `data.x`
- `edge_index`: Edge list → `data.edge_index`
- `edge_attr`: Edge attributes → `data.edge_attr`

These inputs are automatically passed from the `Data` object when calling the model.

**Output:**  
The model returns **node representations (embeddings)**—a tensor with one row per node and one column per output feature.

### Advancing to an Encoder-Decoder Architecture

However, the original GATv2 model is primarily designed to learn **node embeddings**. These are useful for tasks such as node classification but are **not directly applicable to predicting edge attributes** like our target edge weight `tracks`.

Since our objective is to **predict edge values**, using a node-only model is insufficient. To address this, we extend the GATv2 architecture by incorporating a **decoder module** that transforms node embeddings into edge-level predictions.

Our final model follows a typical **encoder-decoder architecture**:

- **Encoder:**  
  We use the GATv2 model as the encoder to compute **informative node embeddings** based on the graph structure, edge attributes, and (if available) node features.

- **Decoder:**  
  As the decoder, we use a **small multilayer perceptron (MLP)** that takes as input the **concatenated embeddings** of each edge's source and target nodes.  
  This MLP outputs a **single scalar value per edge**, which serves as the prediction for the edge attribute `tracks`.
