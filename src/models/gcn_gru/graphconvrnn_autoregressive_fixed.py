import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import networkx as nx
from torch_geometric.utils import from_networkx
import glob
import os
import pandas as pd
import json
import numpy as np
from tsl.metrics.numpy.functional import rmse, mape
import re
from tqdm import tqdm
import joblib
from pathlib import Path
from itertools import product


# --- EarlyStopping Helper Class ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# --- Model Definition ---
class GraphConvRNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, rnn_layers=1):
        super().__init__()
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.gru = nn.GRU(hidden_channels, hidden_channels, rnn_layers, batch_first=True)
        self.edge_mlp = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None, hidden=None):
        x = self.gcn(x, edge_index)
        x = torch.relu(x)
        x_seq = x.unsqueeze(0)
        out_seq, hidden = self.gru(x_seq, hidden)
        node_embeddings = out_seq.squeeze(0)
        source, target = edge_index
        edge_embeddings = (node_embeddings[source] + node_embeddings[target]) / 2
        edge_output = self.edge_mlp(edge_embeddings)
        return edge_output.squeeze(), hidden


# --- Helper function to find scalers ---
def find_scaler_path(scaler_name):
    """
    Find scaler files in the project directory structure.
    Specifically looks in the WebMining_Team6_bikeNetworkAnalysis/src/utils/data_splits/scalers/ directory.
    """
    # Start with the current directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # First try: Look for the standard scaler directory in the project structure
    project_root_candidates = [current_dir] + list(current_dir.parents)
    for root_dir in project_root_candidates:
        if root_dir.name == "WebMining_Team6_bikeNetworkAnalysis" or "WebMining_Team6_bikeNetworkAnalysis" in str(
                root_dir):
            scaler_dir = root_dir / "src" / "utils" / "data_splits" / "scalers"
            potential_path = scaler_dir / scaler_name
            if potential_path.exists():
                print(f"[INFO] Found scaler at standard project path: {potential_path}")
                return str(potential_path)

    # Second try: Look for a scalers directory relative to the current file
    # Navigate up to find models directory and then look for utils/data_splits/scalers
    for parent in current_dir.parents:
        if parent.name == "models":
            scaler_dir = parent.parent / "utils" / "data_splits" / "scalers"
            potential_path = scaler_dir / scaler_name
            if potential_path.exists():
                print(f"[INFO] Found scaler relative to models directory: {potential_path}")
                return str(potential_path)

    # Third try: Exhaustive search within certain depth
    for root_dir in project_root_candidates[:4]:  # Limit search depth
        for potential_dir in root_dir.glob("**/scalers"):
            potential_path = potential_dir / scaler_name
            if potential_path.exists():
                print(f"[INFO] Found scaler by deep search: {potential_path}")
                return str(potential_path)

    # If not found in parent directories, try absolute path
    if os.path.exists(scaler_name):
        return scaler_name

    # If all else fails, try the explicit path mentioned
    explicit_path = Path(
        "WebMining_Team6_bikeNetworkAnalysis") / "src" / "utils" / "data_splits" / "scalers" / scaler_name
    if explicit_path.exists():
        print(f"[INFO] Found scaler at explicit path: {explicit_path}")
        return str(explicit_path)

    raise FileNotFoundError(
        f"Could not find scaler file: {scaler_name}. Searched in WebMining_Team6_bikeNetworkAnalysis/src/utils/data_splits/scalers/ and other locations.")


# Function to load scalers
def load_scalers():
    """Load scalers for inverse transformation only"""
    print("[INFO] Loading scalers for inverse transformation...")

    try:
        # Use the enhanced finder function to locate scalers
        node_scaler_path = find_scaler_path("node_scaler.pkl")
        edge_scaler_path = find_scaler_path("edge_scaler.pkl")
        target_scaler_path = find_scaler_path("target_scaler.pkl")

        node_scaler = joblib.load(node_scaler_path)
        edge_scaler = joblib.load(edge_scaler_path)
        target_scaler = joblib.load(target_scaler_path)

        print(f"[INFO] Scalers successfully loaded from: {os.path.dirname(node_scaler_path)}")
        return node_scaler, edge_scaler, target_scaler
    except Exception as e:
        print(f"[ERROR] Error loading scalers: {e}")
        raise


def build_node_mapping(data_list):
    """
    Build a mapping of node identifiers across graphs to track which nodes are the same.
    This uses spatial coordinates (lat, lon) as a unique identifier.
    Returns:
    - node_mapping: Dictionary mapping (lat, lon) to list of (graph_idx, node_idx)
    - global_node_ids: Dictionary mapping (graph_idx, node_idx) to a unique global node ID
    - local_to_global: List of dictionaries (one per graph) mapping local node idx to global node ID
    """
    # Use lat/lon as unique identifier for nodes
    node_mapping = {}  # (lat, lon) -> list of (graph_idx, node_idx)
    global_node_id_counter = 0
    global_node_ids = {}  # (graph_idx, node_idx) -> global_node_id

    # First pass: build the node mapping
    for graph_idx, data in enumerate(data_list):
        if not hasattr(data, 'x') or data.x is None:
            continue

        # Extract lat/lon from node features
        for node_idx in range(data.x.size(0)):
            try:
                # Assuming first two features are lat/lon
                lat = data.x[node_idx, 0].item()
                lon = data.x[node_idx, 1].item()
                key = (round(lat, 6), round(lon, 6))  # Round to 6 decimal places for precision

                if key not in node_mapping:
                    node_mapping[key] = []
                node_mapping[key].append((graph_idx, node_idx))
            except Exception as e:
                print(f"[WARN] Error extracting coordinates for node {node_idx} in graph {graph_idx}: {e}")

    # Second pass: assign unique global IDs to each unique node
    for coords, instances in node_mapping.items():
        # Assign the same global ID to all instances of this node
        for graph_idx, node_idx in instances:
            global_node_ids[(graph_idx, node_idx)] = global_node_id_counter
        global_node_id_counter += 1

    # Third pass: create local-to-global ID mapping for each graph
    local_to_global = []
    for graph_idx, data in enumerate(data_list):
        if not hasattr(data, 'x') or data.x is None:
            local_to_global.append({})
            continue

        local_map = {}
        for node_idx in range(data.x.size(0)):
            if (graph_idx, node_idx) in global_node_ids:
                local_map[node_idx] = global_node_ids[(graph_idx, node_idx)]
            else:
                # If we couldn't map this node, use a negative ID as fallback
                local_map[node_idx] = -1
        local_to_global.append(local_map)

    print(f"[INFO] Identified {global_node_id_counter} unique nodes across all graphs")
    return node_mapping, global_node_ids, local_to_global


# --- Data Loading ---
def sorted_nicely(l):
    """Sorts the given iterable in the way that humans expect."""

    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s

    def alphanum_key(key):
        return [tryint(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)


def load_graphml(year):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    path = os.path.join(PROJECT_ROOT, "data", "graphml", str(year), "*.graphml")
    print(f"[INFO] Looking for GraphML files at: {path}")

    files = glob.glob(path)
    files = sorted_nicely(files)
    print(f"[INFO] Found {len(files)} GraphML files for year {year}")

    if not files:
        raise FileNotFoundError(f"[ERROR] No GraphML files found for year {year} at {path}")

    data_list = []
    for f in files:
        print(f"[INFO] Loading graph: {f}")
        try:
            G = nx.read_graphml(f)
            # Print the edge attributes to verify structure
            if len(G.edges) > 0:
                first_edge = list(G.edges(data=True))[0]
                print(f"[DEBUG] Edge attributes in file: {first_edge[2].keys()}")

            data = from_networkx(G, group_node_attrs=['lat', 'lon'],
                                 group_edge_attrs=['id', 'year', 'month', 'speed_rel', 'tracks'])

            # Verify the structure of edge_attr
            if data.edge_attr.size(0) > 0:
                print(f"[DEBUG] First few edge attributes: {data.edge_attr[0]}")

            # The 'tracks' attribute is our target
            # Based on GraphML schema, it should be at index 4
            tracks_idx = 4  # This might need adjustment based on debug output
            data.y = data.edge_attr[:, tracks_idx].clone()

            # Remove tracks from edge attributes to avoid leakage
            if data.edge_attr.size(1) > tracks_idx + 1:
                data.edge_attr = torch.cat([
                    data.edge_attr[:, :tracks_idx],
                    data.edge_attr[:, tracks_idx + 1:]
                ], dim=1)
            else:
                data.edge_attr = data.edge_attr[:, :tracks_idx]

            data_list.append(data)

        except Exception as e:
            print(f"[ERROR] Failed to load graph {f}: {e}")
            import traceback
            traceback.print_exc()

    return data_list


def handle_new_nodes(graph, local_to_global, node_feature_cache, mean_node_features):
    """Update features for nodes in the graph using cached features when available"""
    if not hasattr(graph, 'x') or graph.x is None or graph.x.numel() == 0:
        return graph, 0

    updated_count = 0

    for node_idx in range(graph.x.size(0)):
        global_id = local_to_global.get(node_idx, -1)

        # If this node has cached features, use them
        if global_id in node_feature_cache:
            graph.x[node_idx] = node_feature_cache[global_id].to(graph.x.device)
            updated_count += 1
        # Otherwise, use neighborhood aggregation for new nodes
        else:
            # Find all neighboring nodes
            neighbors = []
            for i in range(graph.edge_index.size(1)):
                if graph.edge_index[0, i].item() == node_idx:
                    neighbors.append(graph.edge_index[1, i].item())
                elif graph.edge_index[1, i].item() == node_idx:
                    neighbors.append(graph.edge_index[0, i].item())

            # If we have neighbors with known features, average them
            if neighbors:
                known_neighbors = []
                for neighbor_idx in neighbors:
                    neighbor_global_id = local_to_global.get(neighbor_idx, -1)
                    if neighbor_global_id in node_feature_cache:
                        known_neighbors.append(node_feature_cache[neighbor_global_id])

                if known_neighbors:
                    avg_features = torch.stack(known_neighbors).mean(dim=0).to(graph.x.device)
                    graph.x[node_idx] = avg_features
                    updated_count += 1
                    continue

            # If no useful neighbors, use global mean features
            graph.x[node_idx] = mean_node_features.to(graph.x.device)
            updated_count += 1

    return graph, updated_count


def map_predictions_to_groundtruth(test_graphs, forecast_outputs, target_scaler):
    """Map predictions to ground truth and inverse transform for evaluation"""
    y_true_norm = []
    y_pred_norm = []

    for idx, data in enumerate(tqdm(test_graphs, desc="Mapping forecasts to ground truth")):
        if idx >= len(forecast_outputs):
            print(f"[WARN] No forecast for month {idx + 1}, skipping.")
            continue

        forecast = forecast_outputs[idx].detach().cpu().numpy()

        # Get the ground truth (already normalized)
        if hasattr(data, 'y') and data.y is not None and data.y.numel() > 0:
            y_true = data.y.cpu().numpy()

            # Ensure forecast and ground truth have same shape for comparison
            min_len = min(len(y_true), len(forecast))
            if min_len > 0:
                y_true = y_true[:min_len]
                forecast = forecast[:min_len]

                # Store normalized values
                y_true_norm.append(y_true)
                y_pred_norm.append(forecast)
            else:
                print(f"[WARN] No valid edges to compare for month {idx + 1}")
        else:
            print(f"[WARN] No ground truth for month {idx + 1}")

    # Concatenate all months
    if y_true_norm and y_pred_norm:
        try:
            y_true_normalized = np.concatenate(y_true_norm)
            y_pred_normalized = np.concatenate(y_pred_norm)

            # Only inverse transform for final evaluation
            y_true_original = target_scaler.inverse_transform(y_true_normalized.reshape(-1, 1)).flatten()
            y_pred_original = target_scaler.inverse_transform(y_pred_normalized.reshape(-1, 1)).flatten()

            return y_true_original, y_pred_original, y_true_normalized, y_pred_normalized
        except Exception as e:
            print(f"[ERROR] Failed to process predictions: {e}")

    print("[ERROR] No valid predictions mapped!")
    return np.array([]), np.array([]), np.array([]), np.array([])


def optimize_hyperparameters(train_data, val_data):
    """Find optimal hyperparameters using time-based validation"""
    best_val_loss = float('inf')
    best_params = {}

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device} for hyperparameter optimization")

    # Hyperparameter grid - keep it small for efficiency
    param_grid = {
        'lr': [0.001, 0.0005],
        'weight_decay': [0, 1e-5],
        'hidden_channels': [64, 128],
        'rnn_layers': [1, 2]
    }

    # Generate all combinations of hyperparameters
    param_combinations = list(product(
        param_grid['lr'],
        param_grid['weight_decay'],
        param_grid['hidden_channels'],
        param_grid['rnn_layers']
    ))

    print(f"[INFO] Testing {len(param_combinations)} hyperparameter combinations")

    for lr, wd, hidden_dim, rnn_layers in tqdm(param_combinations, desc="HPO Progress"):
        # Initialize model with current hyperparameters
        model = GraphConvRNN(
            in_channels=train_data[0].num_node_features,
            hidden_channels=hidden_dim,
            out_channels=1,
            rnn_layers=rnn_layers
        ).to(device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Training loop with early stopping
        early_stopper = EarlyStopping(patience=5)

        for epoch in range(1, 21):  # Max 20 epochs for HPO (to save time)
            # Training phase
            model.train()
            total_train_loss = 0

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out, _ = model(data.x, data.edge_index, data.edge_attr)
                loss = nn.MSELoss()(out.view(-1), data.y.view(-1))
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out, _ = model(data.x, data.edge_index, data.edge_attr)
                    val_loss += nn.MSELoss()(out.view(-1), data.y.view(-1)).item()

            val_loss /= len(val_loader)

            if epoch % 5 == 0:
                print(f"  [HPO] lr={lr}, wd={wd}, hidden={hidden_dim}, layers={rnn_layers} | "
                      f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")

            # Check for early stopping
            if early_stopper.step(val_loss):
                print(f"  [HPO] Early stopping at epoch {epoch}")
                break

            # Check if this is the best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {
                    'lr': lr,
                    'weight_decay': wd,
                    'hidden_channels': hidden_dim,
                    'rnn_layers': rnn_layers,
                    'val_loss': val_loss
                }

    print(f"[INFO] Best hyperparameters found: {best_params}")

    # Save best hyperparameters
    best_params_clean = {k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v for k, v in
                         best_params.items()}
    with open('best_hyperparams.json', 'w') as f:
        json.dump(best_params_clean, f, indent=4)

    return best_params


def train_final_model(all_train_data, best_params):
    """Train the final model on all historical data with best hyperparameters"""
    # Check if we have all required parameters, set defaults if missing
    if 'hidden_channels' not in best_params:
        print("[WARN] 'hidden_channels' not found in best_params, using default value of 64")
        best_params['hidden_channels'] = 64

    if 'rnn_layers' not in best_params:
        print("[WARN] 'rnn_layers' not found in best_params, using default value of 1")
        best_params['rnn_layers'] = 1

    if 'lr' not in best_params:
        print("[WARN] 'lr' not found in best_params, using default value of 0.001")
        best_params['lr'] = 0.001

    if 'weight_decay' not in best_params:
        print("[WARN] 'weight_decay' not found in best_params, using default value of 1e-5")
        best_params['weight_decay'] = 1e-5

    # Print the parameters we're using
    print(f"[INFO] Training with parameters: {best_params}")


    # Create dataloader
    train_loader = DataLoader(all_train_data, batch_size=1, shuffle=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model with parameters
    model = GraphConvRNN(
        in_channels=all_train_data[0].num_node_features,
        hidden_channels=best_params['hidden_channels'],
        out_channels=1,
        rnn_layers=best_params['rnn_layers']
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay']
    )

    # Training loop
    print("[INFO] Training final model on all historical data...")

    # Track losses for visualization
    epoch_losses = []

    for epoch in range(1, 101):  # Up to 100 epochs for final model
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, _ = model(data.x, data.edge_index, data.edge_attr)
            loss = nn.MSELoss()(out.view(-1), data.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[INFO] Epoch {epoch}: Training Loss = {avg_loss:.6f}")

    print("[INFO] Final model training completed")

    # Save the model
    torch.save(model.state_dict(), 'final_model.pt')

    # Save training history
    pd.DataFrame({'epoch': range(1, len(epoch_losses) + 1), 'loss': epoch_losses}).to_csv('training_history.csv',
                                                                                          index=False)

    return model


def make_predictions(model, historical_data, test_data, node_feature_cache=None, local_to_global=None,
                     mean_node_features=None):
    """Make single-step predictions for each month in test data"""
    # Load scalers for inverse transformation
    _, _, target_scaler = load_scalers()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare for forecasting
    model.eval()
    forecasts = []

    # If node_feature_cache not provided, initialize it
    if node_feature_cache is None:
        node_feature_cache = {}

    # If mean_node_features not provided, calculate it
    if mean_node_features is None:
        # Calculate mean node features for handling new nodes
        mean_node_features = torch.zeros_like(historical_data[0].x[0])
        node_count = 0

        for data in historical_data:
            if hasattr(data, 'x') and data.x is not None and data.x.numel() > 0:
                mean_node_features += data.x.sum(dim=0)
                node_count += data.x.size(0)

        if node_count > 0:
            mean_node_features /= node_count

    print(f"[INFO] Using {len(historical_data)} historical graphs as context")

    for month_idx in range(len(test_data)):
        print(f"[INFO] Generating forecast for month {month_idx + 1} of 2024...")

        # Reset hidden state for each new forecast
        hidden = None

        # Process all historical context graphs to build initial state
        for ctx_graph in tqdm(historical_data, desc="Processing historical context"):
            ctx_graph = ctx_graph.to(device)
            # Apply handle_new_nodes if we have local_to_global mapping
            if local_to_global is not None:
                ctx_graph, _ = handle_new_nodes(
                    ctx_graph,
                    local_to_global[historical_data.index(ctx_graph)],
                    node_feature_cache,
                    mean_node_features
                )

            with torch.no_grad():
                _, hidden = model(ctx_graph.x, ctx_graph.edge_index, ctx_graph.edge_attr, hidden)

        # Process preceding test months (if any)
        for prev_idx in range(month_idx):
            prev_graph = test_data[prev_idx].to(device)
            # Apply handle_new_nodes if we have local_to_global mapping
            if local_to_global is not None:
                prev_graph, _ = handle_new_nodes(
                    prev_graph,
                    local_to_global[len(historical_data) + prev_idx],
                    node_feature_cache,
                    mean_node_features
                )

            with torch.no_grad():
                _, hidden = model(prev_graph.x, prev_graph.edge_index, prev_graph.edge_attr, hidden)

        # Make prediction for current month
        current_graph = test_data[month_idx].to(device)
        # Apply handle_new_nodes if we have local_to_global mapping
        if local_to_global is not None:
            current_graph, updated_count = handle_new_nodes(
                current_graph,
                local_to_global[len(historical_data) + month_idx],
                node_feature_cache,
                mean_node_features
            )
            print(f"[INFO] Updated features for {updated_count} nodes in current graph")

        with torch.no_grad():
            out, _ = model(current_graph.x, current_graph.edge_index, current_graph.edge_attr, hidden)

        forecasts.append(out.cpu())

    print("[INFO] All single-step forecasts completed")

    # Save forecasts in both normalized and original scale
    normalized_forecasts = [f.numpy() for f in forecasts]
    original_scale_forecasts = []

    for f in normalized_forecasts:
        try:
            # Reshape to column vector for inverse transformation
            orig = target_scaler.inverse_transform(f.reshape(-1, 1)).flatten()
            original_scale_forecasts.append(orig)
        except Exception as e:
            print(f"[ERROR] Failed to inverse transform forecast: {e}")
            original_scale_forecasts.append(np.zeros_like(f))

    # Save normalized forecasts
    df_forecasts_norm = pd.DataFrame(normalized_forecasts).T
    df_forecasts_norm.columns = [f'Month_{i + 1}' for i in range(len(forecasts))]
    df_forecasts_norm.to_csv('forecasts_2024_single_step_normalized.csv', index=False)

    # Save original scale forecasts
    df_forecasts_orig = pd.DataFrame(original_scale_forecasts).T
    df_forecasts_orig.columns = [f'Month_{i + 1}' for i in range(len(forecasts))]
    df_forecasts_orig.to_csv('forecasts_2024_single_step.csv', index=False)

    print("[INFO] Forecasts saved as CSV files")

    return forecasts


def evaluate_forecasts(test_data, forecasts, target_scaler):
    """Evaluate forecasts and calculate metrics"""
    # Map predictions to ground truth and rescale
    y_true_original, y_pred_original, y_true_normalized, y_pred_normalized = map_predictions_to_groundtruth(
        test_data, forecasts, target_scaler
    )

    if len(y_true_original) == 0 or len(y_pred_original) == 0:
        print("[ERROR] No valid predictions to evaluate!")
        return

    # Calculate metrics on original scale
    test_rmse = rmse(y_pred_original, y_true_original)
    test_mape = mape(y_pred_original, y_true_original)
    test_mae = np.mean(np.abs(y_true_original - y_pred_original))

    # Calculate metrics on normalized scale for comparison
    norm_rmse = rmse(y_pred_normalized, y_true_normalized)
    norm_mape = mape(y_pred_normalized, y_true_normalized)
    norm_mae = np.mean(np.abs(y_true_normalized - y_pred_normalized))

    print(f"✅ Test RMSE (original scale): {test_rmse:.4f}")
    print(f"✅ Test MAPE (original scale): {test_mape:.2f}%")
    print(f"✅ Test MAE (original scale): {test_mae:.4f}")
    print(f"✅ Test RMSE (normalized): {norm_rmse:.4f}")
    print(f"✅ Test MAPE (normalized): {norm_mape:.2f}%")
    print(f"✅ Test MAE (normalized): {norm_mae:.4f}")

    # Save test metrics
    test_metrics = {
        'test_rmse_original': float(test_rmse),
        'test_mape_original': float(test_mape),
        'test_mae_original': float(test_mae),
        'test_rmse_normalized': float(norm_rmse),
        'test_mape_normalized': float(norm_mape),
        'test_mae_normalized': float(norm_mae)
    }
    # Save test metrics
    test_metrics = {
        'test_rmse_original': float(test_rmse),
        'test_mape_original': float(test_mape),
        'test_mae_original': float(test_mae),
        'test_rmse_normalized': float(norm_rmse),
        'test_mape_normalized': float(norm_mape),
        'test_mae_normalized': float(norm_mae)
    }

    with open('test_metrics_single_step.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)

    print("✅ Test metrics saved as test_metrics_single_step.json")

    # Create monthly metrics for analysis
    monthly_metrics = []

    for idx, data in enumerate(test_data):
        if idx >= len(forecasts):
            continue

        forecast = forecasts[idx].detach().cpu().numpy()
        y_true = data.y.cpu().numpy()

        # Ensure same length
        min_len = min(len(y_true), len(forecast))
        if min_len > 0:
            # Normalize
            y_true_norm = y_true[:min_len]
            y_pred_norm = forecast[:min_len]

            # Inverse transform
            y_true_orig = target_scaler.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()
            y_pred_orig = target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

            # Calculate metrics
            month_rmse = rmse(y_pred_orig, y_true_orig)
            month_mape = mape(y_pred_orig, y_true_orig)
            month_mae = np.mean(np.abs(y_true_orig - y_pred_orig))

            monthly_metrics.append({
                'month': idx + 1,
                'rmse': float(month_rmse),
                'mape': float(month_mape),
                'mae': float(month_mae),
                'num_edges': min_len
            })

    # Save monthly metrics
    with open('monthly_metrics.json', 'w') as f:
        json.dump(monthly_metrics, f, indent=4)

    print("✅ Monthly metrics saved as monthly_metrics.json")

    # Create visualization data
    if monthly_metrics:
        months = [m['month'] for m in monthly_metrics]
        rmse_values = [m['rmse'] for m in monthly_metrics]

        # Generate CSV for easy plotting
        monthly_df = pd.DataFrame(monthly_metrics)
        monthly_df.to_csv('monthly_performance.csv', index=False)
        print("✅ Monthly performance data saved for visualization")


def main():
    try:
        # Load scalers only for inverse transformation
        _, _, target_scaler = load_scalers()

        # Load data (already normalized)
        data_2021 = load_graphml(2021)
        data_2022 = load_graphml(2022)
        data_2023 = load_graphml(2023)
        test_data_2024 = load_graphml(2024)

        # Build node mapping for tracking nodes across graphs
        print("[INFO] Building node mapping across all graphs...")
        all_data = data_2021 + data_2022 + data_2023 + test_data_2024
        _, _, local_to_global = build_node_mapping(all_data)  # This line computes local_to_global

        # For HPO: Use 2021-2022 for training, part of 2023 for validation
        hpo_train_data = data_2021 + data_2022 + data_2023[:-3]
        hpo_val_data = data_2023[-3:]

        # Check if we already have optimized hyperparameters
        if os.path.exists('best_hyperparams.json'):
            print("[INFO] Loading existing hyperparameters")
            with open('best_hyperparams.json', 'r') as f:
                best_params = json.load(f)
        else:
            # Hyperparameter Optimization Phase
            print("[INFO] Starting hyperparameter optimization")
            best_params = optimize_hyperparameters(hpo_train_data, hpo_val_data)

        # Ensure best_params has all required keys
        if not all(k in best_params for k in ['hidden_channels', 'rnn_layers', 'lr', 'weight_decay']):
            print("[WARN] best_params is missing required keys, setting default values")
            # Set default values
            default_params = {
                'hidden_channels': 64,
                'rnn_layers': 1,
                'lr': 0.001,
                'weight_decay': 1e-5
            }
            # Update best_params with defaults for missing keys
            for key, value in default_params.items():
                if key not in best_params:
                    best_params[key] = value

            print(f"[INFO] Using parameters: {best_params}")

            # Save the updated parameters
            with open('best_hyperparams_with_defaults.json', 'w') as f:
                json.dump(best_params, f, indent=4)

        # Final Model Training: Use ALL historical data with best hyperparameters
        all_train_data = data_2021 + data_2022 + data_2023

        # Initialize node feature cache
        node_feature_cache = {}  # global_node_id -> feature vector

        # Calculate mean node features for new nodes
        mean_node_features = torch.zeros_like(all_train_data[0].x[0])
        node_count = 0

        for data in all_train_data:
            if hasattr(data, 'x') and data.x is not None and data.x.numel() > 0:
                mean_node_features += data.x.sum(dim=0)
                node_count += data.x.size(0)

        if node_count > 0:
            mean_node_features /= node_count

        # Train final model with best hyperparameters on all historical data
        print("[INFO] Training final model on all historical data")
        final_model = train_final_model(all_train_data, best_params)

        # Make predictions for 2024
        print("[INFO] Making predictions for 2024")
        forecasts = make_predictions(
            final_model,
            all_train_data,
            test_data_2024,
            node_feature_cache=node_feature_cache,
            local_to_global=local_to_global,  # Now this is defined
            mean_node_features=mean_node_features
        )

        # Evaluate predictions
        print("[INFO] Evaluating predictions")
        evaluate_forecasts(test_data_2024, forecasts, target_scaler)

        print("[INFO] Pipeline completed successfully!")

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
