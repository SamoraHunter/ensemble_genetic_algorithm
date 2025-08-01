import time
from ml_grid.ga_functions.ga_ann_util import BinaryClassification
import scipy
import numpy as np
from ml_grid.pipeline.torch_binary_classification_method_ga import (
    BinaryClassification,
    TestData,
)
from sklearn import metrics
import torch
import numpy
import pandas as pd
from torch.utils.data import DataLoader
from ml_grid.ga_functions.ga_ensemble_weight_finder_de import (
    get_weighted_ensemble_prediction_de_cython,
)


def super_ensemble_weight_finder_differential_evolution_eval(
    best, ml_grid_object, valid=False
):
    """
    Finds the optimal ensemble weights for a set of models using differential evolution.

    This function evaluates an ensemble of models by fitting each model, generating predictions,
    and then optimizing the weights assigned to each model's predictions to maximize the ensemble's
    performance (AUC score) on a validation or test set. The optimization is performed using
    scipy's differential evolution algorithm.

    Args:
        best (list): A list representing the current best ensemble configuration. Each element should
            be a tuple containing model information and feature columns.
        ml_grid_object (object): An object containing training and test data, as well as configuration
            parameters such as verbosity. Must have attributes:
                - X_train, X_test, y_train, y_test, X_test_orig, y_test_orig, verbose
        valid (bool, optional): If True, use the original test set (validation set) for optimization.
            If False, use the regular test set. Default is False.

    Returns:
        numpy.ndarray: The optimal weights for the ensemble models as determined by differential evolution.

    Raises:
        ValueError: If model fitting or prediction fails due to data shape or type issues.
        Exception: If the differential evolution optimization fails.

    Notes:
        - Handles both scikit-learn style models and custom BinaryClassification (PyTorch) models.
        - Prints warnings and debug information based on the verbosity level of ml_grid_object.
        - Calculates and prints the unweighted ensemble AUC before optimization.
        - Returns the weights that maximize the ensemble's AUC score.
    """
    X_test_orig = ml_grid_object.X_test_orig
    y_test_orig = ml_grid_object.y_test_orig
    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train
    y_test = ml_grid_object.y_test

    # Choose prediction target based on valid parameter
    if valid:
        if ml_grid_object.verbose >= 1:
            print("Finding optimal weights using validation set")
        x_test = X_test_orig.copy()
        y_test_target = y_test_orig.copy()
    else:
        if ml_grid_object.verbose >= 1:
            print("Finding optimal weights using test set")
        x_test = X_test.copy()
        y_test_target = y_test.copy()

    y_test_target = y_test_target.copy()
    if isinstance(y_test_target, pd.Series):
        y_test_target = y_test_target.values

    debug = ml_grid_object.verbose > 11

    if debug:
        print("super_ensemble_weight_finder_differential_evolution, best:")
        print(best)

    model_train_time_warning_threshold = 5

    # Always fit models and generate prediction matrix
    prediction_array = []
    target_ensemble = best[0]

    for i in range(0, len(target_ensemble)):
        feature_columns = list(target_ensemble[i][2])

        existing_columns = [
            col
            for col in feature_columns
            if col in X_train.columns and col in x_test.columns
        ]

        missing_columns = [
            col for col in existing_columns if col not in feature_columns
        ]

        if ml_grid_object.verbose >= 1 and len(missing_columns) >= 1:
            print("Warning: The following columns do not exist in feature_columns:")
            print("\n".join(missing_columns))

        feature_columns = existing_columns.copy()

        if type(target_ensemble[i][1]) is not BinaryClassification:
            model = target_ensemble[i][1]
            if ml_grid_object.verbose >= 2:
                print(f"Fitting model {i+1} for weight optimization")

            try:
                model.fit(X_train[feature_columns], y_train)
                y_pred = model.predict(x_test[feature_columns])
            except ValueError as e:
                print(f"ValueError on fit for model {i+1}: {e}")
                print("feature_columns length:", len(feature_columns))
                print("X_train shape:", X_train.shape, "x_test shape:", x_test.shape)
                raise e

            prediction_array.append(y_pred)

        else:
            # Handle BinaryClassification (PyTorch) models
            if ml_grid_object.verbose >= 2:
                print(f"Handling torch model {i+1} for weight optimization")

            test_data = TestData(torch.FloatTensor(x_test[feature_columns].values))
            test_loader = DataLoader(dataset=test_data, batch_size=1)

            device = torch.device("cpu")
            model = target_ensemble[i][1]
            model.to(device)

            y_hat = model(test_data.X_data.to(device))
            y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()
            y_hat = y_hat.astype(int).flatten()

            if numpy.isnan(y_hat).any():
                print(
                    f"Returning dummy random yhat vector for torch model {i+1}, nan found"
                )
                y_hat = np.random.choice(a=[False, True], size=(len(y_hat),))

            prediction_array.append(y_hat)

    prediction_matrix = np.matrix(prediction_array)
    prediction_matrix = prediction_matrix.astype(float)
    prediction_matrix_raw = prediction_matrix

    # Calculate unweighted ensemble performance
    y_pred_best = []
    for i in range(0, len(prediction_array[0])):
        y_pred_best.append(round(np.mean(prediction_matrix_raw[:, i])))
    auc = metrics.roc_auc_score(y_test_target, y_pred_best)
    print("Unweighted ensemble AUC: ", auc)

    bounds = [(0, 1) for x in range(0, len(best[0]))]

    start = time.time()
    try:
        de = scipy.optimize.differential_evolution(
            get_weighted_ensemble_prediction_de_cython,
            bounds,
            args=((prediction_matrix_raw, y_test_target)),
            strategy="best1bin",
            maxiter=20,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=None,
            callback=None,
            disp=False,
            polish=True,
            init="latinhypercube",
            atol=0,
            updating="immediate",
            workers=4,
            constraints=(),
            x0=None,
            # integrality=None,
            # vectorized=False
        )
    except Exception as e:
        print("Failed on s e wf DE", e)
        print(prediction_matrix_raw, y_test_target)
        raise e

    score = 1 - de.fun
    optimal_weights = de.x

    end = time.time()
    model_train_time = int(end - start)
    if debug:
        if model_train_time > model_train_time_warning_threshold:
            print(
                "Warning long DE weights train time, ",
                model_train_time,
                model_train_time_warning_threshold,
            )

    print("best weighted score: ", score, "difference:", score - auc)
    # print("best weights", optimal_weights, optimal_weights.shape)
    return optimal_weights
