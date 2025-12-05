import itertools
import logging
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ml_grid.ga_functions.ga_ann_util import (
    BinaryClassification,
    TestData,
    TrainData,
    binary_acc,
    get_free_gpu,
)
from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.model_methods_ga import store_model
from ml_grid.util.validate_param_methods import hidden_layer_size

logger = logging.getLogger("ensemble_ga")


def predict_with_fallback(
    model: nn.Module, X_batch: torch.Tensor, y_batch: torch.Tensor
) -> torch.Tensor:
    """Predicts using the model, with a fallback to zero logits on error.

    This function attempts to get a prediction from the model. If any exception
    occurs during the forward pass, it catches the error, prints a warning,
    and returns a tensor of zero logits with the correct shape, allowing
    the training loop to continue without crashing.

    Args:
        model (nn.Module): The PyTorch model to use for prediction.
        X_batch (torch.Tensor): The input data batch.
        y_batch (torch.Tensor): The target data batch (used for shape inference).

    Returns:
        torch.Tensor: The model's output tensor or a tensor of zero logits
        if an exception occurred.
    """
    try:
        y_pred = model(X_batch)
    except Exception as e:
        logger.warning(f"Model prediction failed with error: {e}. Using fallback.")
        # Fallback: return zero logits with the correct output shape.
        # BCEWithLogitsLoss expects shape (batch_size, 1).
        y_pred = torch.zeros(
            y_batch.unsqueeze(1).shape, device=X_batch.device, dtype=X_batch.dtype
        )

    return y_pred


def Pytorch_binary_class_ModelGenerator(
    ml_grid_object: Any, local_param_dict: Dict
) -> Tuple[float, BinaryClassification, List[str], int, float, np.ndarray]:
    """Generates, trains, and evaluates a PyTorch-based binary classification ANN.

    This function performs a single trial of training and evaluating a custom
    PyTorch Artificial Neural Network (ANN). It uses a random search approach
    for hyperparameter tuning.

    The process includes:
    1.  Applying ANOVA-based feature selection.
    2.  Optionally scaling the data using StandardScaler.
    3.  Preparing PyTorch DataLoader for training.
    4.  Randomly sampling hyperparameters (e.g., batch size, layer size,
        dropout, epochs, learning rate) from a predefined search space.
    5.  Selecting an available GPU or defaulting to CPU.
    6.  Training the PyTorch `BinaryClassification` model.
    7.  Evaluating the model's performance on the test set using Matthews
        Correlation Coefficient (MCC) and ROC AUC score.
    8.  Handling potential NaN predictions with a random fallback.
    9.  Optionally storing the trained model and its metadata.

    Args:
        ml_grid_object (Any): An object containing the project's data (e.g.,
            X_train, y_train, X_test, y_test) and configuration settings.
        local_param_dict (Dict): A dictionary of local parameters for this
            specific model run, which may include 'scale'.

    Returns:
        A tuple containing the following elements:
            - mccscore (float): The Matthews Correlation Coefficient.
            - model (BinaryClassification): The trained PyTorch model object.
            - feature_names (List[str]): A list of feature names used for training.
            - model_train_time (int): The model training time in seconds.
            - auc_score (float): The ROC AUC score.
            - y_pred (np.ndarray): The model's predictions on the test set.

    """
    from ml_grid.util.global_params import global_parameters

    global_parameter_val = global_parameters()

    verbose = global_parameter_val.verbose
    store_base_learners = ml_grid_object.global_params.store_base_learners
    scale = ml_grid_object.local_param_dict.get("scale")

    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train
    y_test = ml_grid_object.y_test

    debug = ml_grid_object.verbose >= 1

    start = time.time()

    X_train, X_test = feature_selection_methods_class(
        ml_grid_object
    ).get_featured_selected_training_data(method="anova")

    # Capture column names and length before potential scaling
    feature_names = list(X_train.columns)
    num_features = len(feature_names)

    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()

    if scale:
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train_np)
        X_test_np = scaler.transform(X_test_np)

    train_data = TrainData(
        torch.FloatTensor(X_train_np), torch.FloatTensor(y_train.to_numpy())
    )
    test_data = TestData(torch.FloatTensor(X_test_np))

    # Initialise global parameter space----------------------------------------------------------------

    parameter_space = {
        "column_length": [num_features],
        #'epochs': [50, 200],
        "hidden_layer_size": [
            max(2, int(X_train.shape[0] / 100)),
            max(2, int(X_train.shape[0] / 200)),
        ],
        #'learning_rate': lr_space,
        #'learning_rate': [0.1, 0.001, 0.0005, 0.0001],
        "deep_layers_1": [2, 4, 8, 16, 32],
        "dropout_val": [0.1, 0.01, 0.001],
    }
    # logger.debug("parameter_space", parameter_space)

    additional_grid = {
        "epochs": [10, 50, 100],
        "learning_rate": [0.1, 0.001, 0.0005, 0.0001],
    }
    size_test = []
    # Loop over al grid search combinations
    for values in itertools.product(*additional_grid.values()):
        point = dict(zip(additional_grid.keys(), values))
        # merge the general settings
        settings = {**point}
        # logger.debug(settings)
        size_test.append(settings)

    # logger.debug(len(size_test))

    # Select a random sample from the global parameter space
    sample_parameter_space = {}
    for key in parameter_space.keys():
        sample_parameter_space[key] = random.choice(parameter_space.get(key))

    sample_parameter_space = hidden_layer_size(sample_parameter_space)

    additional_param_sample = random.choice(size_test)

    additional_param_sample = {}
    for key in additional_grid.keys():
        additional_param_sample[key] = random.choice(additional_grid.get(key))

    logger.debug(sample_parameter_space)
    logger.debug(additional_param_sample)

    free_gpu = str(get_free_gpu(ml_grid_object))

    # os.environ["CUDA_VISIBLE_DEVICES"]=free_gpu

    device = torch.device(f"cuda:{free_gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=sample_parameter_space["hidden_layer_size"],
        shuffle=True,
        drop_last=True,
    )
    # test_loader = DataLoader(dataset=test_data, batch_size=1)

    # fit model with random sample of global parameter space
    model = BinaryClassification(**sample_parameter_space)

    model.to(device)
    # logger.debug(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=additional_param_sample["learning_rate"]
    )
    model.train()
    for e in range(1, additional_param_sample["epochs"] + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # y_pred = model(X_batch)
            y_pred = predict_with_fallback(
                model=model, X_batch=X_batch, y_batch=y_batch
            )

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    if ml_grid_object.verbose > 2:
        logger.info(
            f"Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}"
        )

    y_pred_fallback = np.random.randint(2, size=len(y_test))

    try:
        with torch.no_grad():
            model.eval()
            y_pred_tensor = model(test_data.X_data.to(device))
            y_pred = torch.round(torch.sigmoid(y_pred_tensor)).cpu().numpy().flatten()
    except Exception as e:
        if verbose >= 1:
            logger.error(e)
            logger.warning("Returning random label vector")
        y_pred = y_pred_fallback

    # Check for and replace any NaN values before returning
    if np.isnan(y_pred).any():
        logger.warning(
            "Warning: NaN values detected in predictions. Replacing them with random 0 or 1."
        )
        y_pred = np.nan_to_num(y_pred, nan=np.random.randint(2)).astype(
            int  # type: ignore
        )  # Replaces nan with random 0 or 1

    mccscore = matthews_corrcoef(y_test, y_pred)

    auc_score = round(metrics.roc_auc_score(y_test, y_pred), 4)

    end = time.time()
    model_train_time = int(end - start)
    if debug:
        debug_base_learner(model, mccscore, X_train, auc_score, model_train_time)

    if store_base_learners:
        try:
            store_model(
                ml_grid_object,
                local_param_dict,
                mccscore,
                model,  # Note: Storing a PyTorch model might require special handling
                list(X_train.columns),
                model_train_time,
                auc_score,
                y_pred,
                model_type="torch",
            )
        except Exception as e:
            logger.error(e)

    torch.cuda.empty_cache()  # exp

    return (mccscore, model, list(X_train.columns), model_train_time, auc_score, y_pred)
