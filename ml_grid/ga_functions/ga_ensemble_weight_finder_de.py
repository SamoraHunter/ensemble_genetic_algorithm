import time
from ml_grid.ga_functions.ga_de_weight_method import (
    get_weighted_ensemble_prediction_de_y_pred_valid,
)
from ml_grid.ga_functions.ga_unweighted import get_best_y_pred_unweighted
from ml_grid.pipeline.evaluate_methods_ga import (
    get_weighted_ensemble_prediction_de_cython,
)

import numpy
import numpy as np
import pandas as pd
import scipy
import torch
import tqdm
from ml_grid.pipeline import torch_binary_classification_method_ga
from ml_grid.pipeline.torch_binary_classification_method_ga import (
    BinaryClassification,
    TestData,
)

from ml_grid.util.global_params import global_parameters
from sklearn import metrics

import numpy as np
import torch
import numpy
from torch.utils.data import Dataset, DataLoader

from numpy.linalg import norm


# redundant? weights only derived from xtrain, weight vec is size of ensemble not train set


# Only get weights from xtrain/ytrain, never get weights from xtest y test. Use weights on x_validation yhat to compare to ytrue_valid
def super_ensemble_weight_finder_differential_evolution(
    best, ml_grid_object, valid=False
):
    X_test_orig = ml_grid_object.X_test_orig
    y_test_orig = ml_grid_object.y_test_orig
    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train
    y_test = ml_grid_object.y_test

    debug = ml_grid_object.debug  # set in data?

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

    # Set valid set after #don't set valid ever
    # print("super_ensemble_weight_finder_differential_evolution", valid)
    #     if(valid):
    #         x_test = X_test_orig.copy()
    #         y_test = y_test_orig.copy()

    bounds = [(0, 1) for x in range(0, len(best[0]))]

    start = time.time()
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
