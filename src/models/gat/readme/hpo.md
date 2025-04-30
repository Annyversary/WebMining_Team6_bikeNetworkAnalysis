## Hyperparamter Optimization 

### Random Search

This section performs a randomized hyperparameter search within a predefined search space to optimize a `GATv2EdgePredictor` model for edge attribute regression.

For each trial, a random configuration is sampled. A model is then instantiated using this configuration. It is trained on the training set for up to `num_epochs`, using the Adam optimizer and Mean Squared Error (MSE) loss. The prediction task targets the specific edge attribute `tracks` (`target_idx = 4`).

**Early Stopping:** During training, the model is evaluated on the validation set every 50 epochs. If the validation loss does not improve for a defined number of evaluations (`early_stopping_patience`), training is stopped early to prevent overfitting.

**Output:** After all trials are completed, the best-performing model (based on validation loss) and its corresponding hyperparameter configuration are saved:
- Model weights: `hpo_models/best_model_overall.pth`
- Configuration file: `hpo_models/best_config_overall.json`

The `hpo_models/` directory is automatically created if it does not already exist, to store the results. Random seeds are fixed using `random.seed(42)` and `torch.manual_seed(42)` to ensure reproducibility across runs.

### Defining the Hyperparameter Search Space and Training Settings

The hyperparameter search space is defined as a set of possible values for several key hyperparameters, including:
- the learning rate (`lr`),
- the number of hidden units in intermediate layers (`hidden_channels`),
- the size of the output features (`out_channels`),
- and the number of attention heads (`heads`) used in the GATv2 model.

This search space can be expanded or modified depending on the model's requirements. For example:
- **Optimizer**: We currently use Adam, but alternatives like SGD or AdamW could also be tested.
- **Loss function**: MSELoss is used here, but other functions such as L1Loss could be considered.
- **Dropout**: This could be applied within GATv2 or in the edge-level MLP, although it is not used in the current implementation.
- **Batch size**: Currently, training is performed on the entire graph (no mini-batching). However, support for mini-batching could be implemented.

In addition to defining the search space, the training and validation settings are specified as follows:
- The number of hyperparameter optimization (HPO) trials (`num_trials`) is set to 20.
- Each model is trained for a maximum of 1000 epochs (`num_epochs`).
- Early stopping is applied with a patience of X (`early_stopping_patience`), which corresponds to X × 50 training epochs without improvement in the validation loss (since evaluation occurs every 50 epochs).
- The minimum change required for a validation loss to be considered an improvement is defined by `min_delta` (set to 0.01). 