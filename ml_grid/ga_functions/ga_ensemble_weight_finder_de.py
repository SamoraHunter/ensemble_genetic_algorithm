import time
from ml_grid.ga_functions.ga_ann_util import normalize
import numpy as np
import pandas as pd
import scipy
from ml_grid.util.global_params import global_parameters
from sklearn import metrics
from numpy.linalg import norm
from sklearn import metrics

round_v = np.vectorize(round)


def get_weighted_ensemble_prediction_de_cython(weights, prediction_matrix_raw, y_test):
    """
    Computes the weighted ensemble prediction for a set of models using the provided weights,
    and evaluates the prediction using ROC AUC score.
    Args:
        weights (array-like): Array of weights for each model in the ensemble.
        prediction_matrix_raw (array-like): 2D array where each row corresponds to the predictions
            from a single model for all test samples.
        y_test (array-like): Ground truth labels for the test samples.
    Returns:
        float: 1 minus the ROC AUC score of the weighted ensemble prediction.
    Raises:
        Exception: If an error occurs during the computation of the ROC AUC score, prints
            debugging information and re-raises the exception.
    """

    clean_prediction_matrix = prediction_matrix_raw.copy()
    weights = normalize(weights)

    weighted_prediction_matrix_array = (
        np.array(clean_prediction_matrix) * weights[:, None]
    )
    collapsed_weighted_prediction_matrix_array = weighted_prediction_matrix_array.sum(
        axis=0
    )

    y_pred_best = round_v(collapsed_weighted_prediction_matrix_array)
    try:
        auc = metrics.roc_auc_score(y_test, y_pred_best)
        score = auc
        return 1 - score
    except Exception as e:
        print(y_test)
        print(y_pred_best)
        print(type(y_test))
        print(type(y_pred_best))
        raise e


# Only get weights from xtrain/ytrain, never get weights from xtest y test. Use weights on x_validation yhat to compare to ytrue_valid
def super_ensemble_weight_finder_differential_evolution(
    best, ml_grid_object, valid=False
):
    """
    Finds the optimal ensemble weights for a set of models using Differential Evolution optimization.

    This function takes a list of models (with their predictions) and a machine learning grid object,
    and uses the Differential Evolution algorithm to find the set of weights that maximize the ensemble's
    ROC AUC score on the test set.

    Args:
        best (list): A list containing the best ensemble models and their associated data. Each element
            should contain model information, including predictions on the test set at index 5.
        ml_grid_object (object): An object containing training and test data, as well as other relevant
            attributes (e.g., X_train, X_test, y_train, y_test, X_test_orig, y_test_orig, verbose).
        valid (bool, optional): Unused parameter, kept for compatibility. Defaults to False.

    Returns:
        np.ndarray: The optimal weights for the ensemble models as determined by Differential Evolution.

    Raises:
        Exception: If the Differential Evolution optimization fails, the exception is printed and re-raised.

    Notes:
        - The function prints the unweighted ensemble AUC and the best weighted score.
        - Uses `get_weighted_ensemble_prediction_de_cython` as the objective function for optimization.
        - Assumes that the predictions in `best` are aligned with `y_test`.
        - The function is designed for binary classification tasks (uses ROC AUC).
    """
    X_test_orig = ml_grid_object.X_test_orig
    y_test_orig = ml_grid_object.y_test_orig
    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train
    y_test = ml_grid_object.y_test

    y_test = y_test.copy()  # WRITEABLE error fix?
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    # debug = ml_grid_object.debug  # set in data? ????

    debug = ml_grid_object.verbose > 11

    if debug:
        print("super_ensemble_weight_finder_differential_evolution, best:")
        print(best)

    model_train_time_warning_threshold = 5
    # Get prediction matrix:
    prediction_array = []
    target_ensemble = best[0]
    for i in range(0, len(target_ensemble)):
        # For model i, predict it's x_test
        # feature_columns = target_ensemble[i][2]
        y_pred = target_ensemble[i][5]
        # print(y_pred.shape)

        prediction_array.append(y_pred)

    prediction_matrix = np.matrix(prediction_array)
    # print(prediction_matrix.shape)
    prediction_matrix = prediction_matrix.astype(float)
    prediction_matrix_raw = prediction_matrix
    y_pred_best = []
    for i in range(0, len(prediction_array[0])):
        y_pred_best.append(round(np.mean(prediction_matrix_raw[:, i])))
    auc = metrics.roc_auc_score(y_test, y_pred_best)
    print("Unweighted ensemble AUC: ", auc)

    bounds = [(0, 1) for x in range(0, len(best[0]))]

    start = time.time()
    try:
        de = scipy.optimize.differential_evolution(
            get_weighted_ensemble_prediction_de_cython,
            bounds,
            args=((prediction_matrix_raw, y_test)),
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
        print(prediction_matrix_raw, y_test)
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
