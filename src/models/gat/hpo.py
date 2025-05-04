import os
import random
import json
import torch
import joblib

from gatv2 import GATv2EdgePredictor
from sklearn.metrics import mean_squared_error, root_mean_squared_error

def perform_hpo(train_data, val_data, search_space, num_trials=3, num_epochs=1000, early_stopping_patience=1, min_delta=0.01, project_root=os.path.join(os.path.dirname(__file__), "..", "..", "..")):
    """
    Performs random hyperparameter optimization for a GATv2 edge predictor model.

    Parameters:
    -----------
    train_data : torch_geometric.data.Data
        The training graph data.
    val_data : torch_geometric.data.Data
        The validation graph data.
    search_space : dict
        Dictionary specifying possible values for hyperparameters (lr, hidden_channels, etc.).
    num_trials : int, optional (default=3)
        Number of random hyperparameter configurations to try.
    num_epochs : int, optional (default=1000)
        Maximum number of training epochs per trial.
    early_stopping_patience : int, optional (default=1)
        Number of evaluations with no improvement before early stopping.
    min_delta : float, optional (default=0.01)
        Minimum change in the monitored quantity to qualify as an improvement. 

    Returns:
    --------
    None
        Saves the best model and configuration to disk as 'best_model_overall.pth' and 'best_config_overall.json'.
    """

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Create directory for saving models and configs
    model_dir = os.path.join(project_root, "src", "models", "gat", "hpo_models")
    os.makedirs(model_dir, exist_ok=True)

    best_overall_val_loss = float("inf")
    best_config = None

    # Begin hyperparameter search loop
    for trial in range(num_trials):
        # Randomly sample a configuration from the search space
        config = {
            "lr": random.choice(search_space["lr"]),
            "hidden_channels": random.choice(search_space["hidden_channels"]),
            "out_channels": random.choice(search_space["out_channels"]),
            "heads": random.choice(search_space["heads"]),
        }

        print(f"\n Trial {trial+1}/{num_trials} | Config: {config}")

        # Instantiate model with current configuration
        model = GATv2EdgePredictor(
            in_channels=train_data.num_node_features,
            hidden_channels=config["hidden_channels"],
            out_channels=config["out_channels"],
            edge_dim=train_data.edge_attr.shape[1],
            heads=config["heads"]
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        criterion = torch.nn.MSELoss()

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        # Training loop
        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad()

            pred = model(train_data)
            target = train_data.y
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            # Print training loss periodically
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Training Loss: {loss.item():.4f}")

            # === Evaluate on validation set every 50 epochs ===
            if epoch % 50 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(val_data)
                    val_target = val_data.y

                    # Compute MSE on validation set using scaled values
                    val_mse = mean_squared_error(val_target.cpu().numpy(), val_pred.cpu().numpy())

                    # Compute RMSE in original scale
                    scaler_path = os.path.join(project_root, "src", "utils", "data_splits", "scalers", "2021_to_2023", "target_scaler.pkl")
                    y_scaler = joblib.load(scaler_path)

                    val_pred_orig = y_scaler.inverse_transform(val_pred.cpu().numpy())
                    val_target_orig = y_scaler.inverse_transform(val_target.cpu().numpy())
                    val_rmse = root_mean_squared_error(val_target_orig, val_pred_orig)
                    val_loss = val_mse


                print(f"Epoch {epoch:03d} | Validation Loss: {val_mse:.4f} || RMSE (original scale): {val_rmse:.2f}")

                # Save the model if it improves the best validation RMSE
                if best_val_loss - val_rmse > min_delta:
                    best_val_loss = val_rmse
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                # Apply early stopping if no improvement
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"Trial {trial+1} complete | Best Validation RMSE: {best_val_loss:.2f}")

        # Update global best model and config if current trial is better
        if best_val_loss < best_overall_val_loss:
            best_overall_val_loss = best_val_loss
            best_config = {
                **config,               # original hyperparameters
                "trial": trial + 1,     # current trial number (1-based)
                "val_loss": best_val_loss
            }
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model_overall.pth"))
            with open(os.path.join(model_dir, "best_config_overall.json"), "w") as f:
                json.dump(best_config, f, indent=2)

        # Clean up memory
        del model, optimizer, pred, target, val_pred, val_target, loss, val_loss
        import gc
        gc.collect()

    print("\nRandom search completed.")
    print(f"Best configuration: {best_config}")
    print(f"Best validation loss: {best_overall_val_loss:.4f}")


# Define hyperparameter search space (can be extended or modified)
search_space = {
    "lr": [1e-1, 1e-2, 1e-3],                    # Learning rates (optional: 1e-4)
    "hidden_channels": [8, 16, 32],              # Hidden layer sizes (optional: 64)
    "out_channels": [8, 16],                     # Output feature sizes from GNN (optional: 32)
    "heads": [1, 2],                             # Number of attention heads in GATv2 (optional: 4)
    # "dropout": [0.0, 0.1, 0.3, 0.5],           # Dropout
}

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
train_data_path = os.path.join(project_root, "data", "data_splits", "2021_to_2023_split", "train_data.pt")
val_data_path = os.path.join(project_root, "data", "data_splits", "2021_to_2023_split", "val_data.pt")

train_data = torch.load(train_data_path, weights_only=False)
val_data = torch.load(val_data_path, weights_only=False)

# Define number of HPO trials and training settings
perform_hpo(train_data, val_data, search_space, num_trials=1, num_epochs=3000, early_stopping_patience=3, min_delta = 0.1, project_root=project_root)
