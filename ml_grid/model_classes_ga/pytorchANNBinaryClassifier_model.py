import itertools
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn import metrics
from ml_grid.ga_functions.ga_ann_util import (
    BinaryClassification,
    TestData,
    TrainData,
    binary_acc,
    get_free_gpu,
)
from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.model_methods_ga import store_model
from sklearn.preprocessing import StandardScaler

from ml_grid.util.validate_param_methods import hidden_layer_size


# def predict_with_fallback(model, X_batch):
#     try:
#         y_pred = model(X_batch)
#         return y_pred
#     except:
#         # If an exception occurs (e.g., model prediction fails), generate a random binary vector
#         random_binary_vector = np.random.randint(2, size=X_batch.shape[0])
#         return random_binary_vector


def predict_with_fallback(model, X_batch, y_batch):
    """
    Predict using the model with fallback mechanism.

    Parameters:
    model (object): The model used for prediction.
    X_batch (object): The input data batch.
    y_batch (object): The target data batch.

    Returns:
    object: The predicted output or a random binary vector in case of an exception.
    """

    # Use model prediction if possible
    try:
        y_pred = model(X_batch)

    # If prediction fails, generate a random binary vector
    except Exception as e:
        print(e)
        print("Failed ypred fallback")
        print("X_batch shape,", X_batch.shape)
        print("Y_batch.shape", y_batch.shape)
        # print("Y_pred.shape", y_pred.shape, type(y_pred), )
        raise e

        y_pred = torch.randint(2, size=X_batch.shape, device=X_batch.device)

    return y_pred


def Pytorch_binary_class_ModelGenerator(ml_grid_object, local_param_dict):
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
        ml_grid_object: An object containing the project's data (e.g.,
            X_train, y_train, X_test, y_test) and configuration settings.
        local_param_dict (dict): A dictionary of local parameters for this
            specific model run, which may include 'scale'.

    Returns:
        tuple: A tuple containing mccscore (float), the trained model object,
        a list of feature names, the model training time (int), the
        auc_score (float), and the predictions (np.ndarray).

    """
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

    if scale == False:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    train_data = TrainData(
        torch.FloatTensor(X_train.to_numpy()), torch.FloatTensor(y_train.to_numpy())
    )
    test_data = TestData(torch.FloatTensor(X_test.to_numpy()))

    # Initialise global parameter space----------------------------------------------------------------

    # print("ANN binary Xtrain")
    # print(X_train)
    # print(type(X_train))
    # print(int(X_train.shape[0]))

    parameter_space = {
        "column_length": [len(X_train.columns)],
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
    # print("parameter_space", parameter_space)

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
        # print(settings)
        size_test.append(settings)

    # print(len(size_test))

    # Select a random sample from the global parameter space
    sample_parameter_space = {}
    for key in parameter_space.keys():
        sample_parameter_space[key] = random.choice(parameter_space.get(key))

    sample_parameter_space = hidden_layer_size(sample_parameter_space)

    additional_param_sample = random.choice(size_test)

    additional_param_sample = {}
    for key in additional_grid.keys():
        additional_param_sample[key] = random.choice(additional_grid.get(key))

    if ml_grid_object.verbose > 0:
        print(sample_parameter_space)

        print(additional_param_sample)

    free_gpu = str(get_free_gpu(ml_grid_object))

    # os.environ["CUDA_VISIBLE_DEVICES"]=free_gpu

    device = torch.device(f"cuda:{free_gpu}" if torch.cuda.is_available() else "cpu")
    if ml_grid_object.verbose > 0:
        print(device)

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
    # print(model)

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
            # print(X_batch.shape)
            # print(type(X_batch)) #torch torch.Tensor
            try:
                y_pred = predict_with_fallback(
                    model=model, X_batch=X_batch, y_batch=y_batch
                )

            except Exception as e:
                print(e)
                print("Failed ypred fallback")
                print("X_batch shape,", X_batch.shape)
                print("Y_batch.shape", y_batch.shape)
                print(
                    "Y_pred.shape",
                    y_pred.shape,
                    type(y_pred),
                )
                raise e

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    if ml_grid_object.verbose > 2:
        print(
            f"Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}"
        )

    para_str = (
        str(settings)
        .replace("'", "_")
        .replace(":", "_")
        .replace(",", "_")
        .replace("{", "_")
        .replace("}", "_")
        .replace(" ", "_")
    ).replace("__", "_")

    try:
        y_pred = model(test_data.X_data.to(device))

        y_pred = torch.round(torch.sigmoid(y_pred)).cpu().detach().numpy().flatten()

    except ValueError as e:
        if verbose >= 1:
            print(e)
            print("Returning random label vector")
            X_test_length = len(X_test)

            y_pred = np.random.randint(2, size=X_test_length)

    # Check for and replace any NaN values before returning
    if np.isnan(y_pred).any():
        print(
            "Warning: NaN values detected in predictions. Replacing them with random 0 or 1."
        )
        y_pred = np.nan_to_num(y_pred, nan=np.random.randint(2)).astype(
            int
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
                model,
                list(X_train.columns),
                model_train_time,
                auc_score,
                y_pred,
                model_type="torch",
            )
        except Exception as e:
            print(e)

    torch.cuda.empty_cache()  # exp

    return (mccscore, model, list(X_train.columns), model_train_time, auc_score, y_pred)
