import logging
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy
from numpy.linalg import norm
from sklearn import metrics

from ml_grid.ga_functions.ga_ann_weight_methods import get_y_pred_ann_torch_weighting
from ml_grid.ga_functions.ga_de_weight_method import (
    get_weighted_ensemble_prediction_de_y_pred_valid,
)
from ml_grid.ga_functions.ga_ensemble_weight_finder_de import find_ensemble_weights_de
from ml_grid.ga_functions.ga_unweighted import (
    get_unweighted_ensemble_predictions,
)
from ml_grid.util.ensemble_diversity_methods import (
    apply_diversity_penalty,
    measure_diversity_wrapper,
)
from ml_grid.util.global_params import global_parameters

logger = logging.getLogger("ensemble_ga")


def get_y_pred_resolver(
    ensemble: List, ml_grid_object: Any, valid: bool = False
) -> Union[List, np.ndarray]:
    """Resolves and generates predictions for an ensemble during GA training.

    This function acts as a dispatcher, calling the appropriate prediction
    generation function (`unweighted`, `de` for Differential Evolution, or `ann`
    for an Artificial Neural Network) based on the 'weighted' parameter in the
    `local_param_dict`.

    These resolver functions are used during the genetic algorithm's evaluation
    phase. They typically use pre-calculated predictions from the base learners
    to quickly evaluate an ensemble's fitness without re-training models.

    Args:
        ensemble: A list containing the ensemble configuration. In DEAP, this
            is often a list containing the actual list of base learners.
        ml_grid_object: The main experiment object, which holds data splits
            (X_test, y_test, X_test_orig, y_test_orig) and configuration
            parameters (`local_param_dict`).
        valid: A boolean flag indicating the dataset to use for prediction.
            If True, predictions are generated for the validation set
            (`X_test_orig`). If False, predictions are for the test set
            (`X_test`). Defaults to False.

    Returns:
        The final ensemble predictions, as either a list or a NumPy array.

    Raises:
        Exception: Propagates exceptions from underlying prediction functions
            if an error occurs during prediction generation.
    """
    if ml_grid_object.verbose >= 1:
        logger.info("get_y_pred_resolver")
        logger.info(ensemble)
    local_param_dict = ml_grid_object.local_param_dict
    X_test_orig = ml_grid_object.X_test_orig
    y_test_orig = ml_grid_object.y_test_orig

    if ml_grid_object.verbose >= 2:
        logger.info("Starting get_y_pred_resolver function...")
        logger.info("local_param_dict: %s", local_param_dict)
        logger.info("X_test_orig shape: %s", X_test_orig.shape)
        logger.info("y_test_orig shape: %s", y_test_orig.shape)

    if (
        local_param_dict.get("weighted") is None
        or local_param_dict.get("weighted") == "unweighted"
    ):
        if ml_grid_object.verbose >= 1:
            logger.info("Using unweighted ensemble prediction...")
        try:
            y_pred = get_unweighted_ensemble_predictions(
                ensemble, ml_grid_object, valid=valid
            )
        except Exception as e:
            logger.error(
                "exception on y_pred = get_unweighted_ensemble_predictions(ensemble, ml_grid_object, valid=valid)"
            )
            logger.error(ensemble)
            logger.error("valid: %s", valid)
            raise e
    elif local_param_dict.get("weighted") == "de":
        if ml_grid_object.verbose >= 1:
            logger.info("Using DE weighted ensemble prediction...")
        y_pred = get_weighted_ensemble_prediction_de_y_pred_valid(
            ensemble,
            find_ensemble_weights_de(ensemble, ml_grid_object, valid=valid),
            ml_grid_object,
            valid=valid,
        )
        if ml_grid_object.verbose >= 2:
            logger.info("DE weighted y_pred shape: %s", y_pred.shape)
    elif local_param_dict.get("weighted") == "ann":
        if ml_grid_object.verbose >= 1:
            logger.info("Using ANN weighted ensemble prediction...")
        y_pred = get_y_pred_ann_torch_weighting(ensemble, ml_grid_object, valid=valid)

    return y_pred


def evaluate_weighted_ensemble_auc(
    individual: List, ml_grid_object: Any
) -> Tuple[float, ...]:
    """The main fitness evaluation function for the genetic algorithm.

    This function is registered with DEAP as the 'evaluate' operator. It takes
    an individual (an ensemble), calculates its performance, and returns a
    fitness value.

    The process includes:
    1.  Generating predictions for the ensemble using `get_y_pred_resolver`.
    2.  Calculating performance metrics (AUC, MCC, F1, etc.).
    3.  Calculating the diversity of the ensemble.
    4.  Applying a penalty to the performance metrics based on the diversity score
        if specified in the configuration.
    5.  Logging the detailed results of the evaluation to a CSV file.
    6.  Returning the final fitness score (either AUC or diversity-penalized AUC)
        as a tuple, as required by DEAP.

    Args:
        individual: The individual to be evaluated, which is a list representing an ensemble.
        ml_grid_object: The main experiment object, containing data and configurations.

    Returns:
        A tuple containing a single float value representing the fitness of the
        individual, as required by DEAP.
    """
    if ml_grid_object.verbose >= 1:
        logger.info("evaluate_weighted_ensemble_auc: individual")
        logger.info(individual)

    global_params = global_parameters()

    ml_grid_object = ml_grid_object

    verbose = ml_grid_object.global_params.verbose

    y_test = ml_grid_object.y_test

    local_param_dict = ml_grid_object.local_param_dict

    file_path = ml_grid_object.logging_paths_obj.log_folder_path

    original_feature_names = ml_grid_object.original_feature_names

    log_store_dataframe_path = global_params.log_store_dataframe_path

    y_pred = get_y_pred_resolver(individual, ml_grid_object, valid=False)
    # should write mcc parallel instead of auc also, need pair functions
    try:
        auc = metrics.roc_auc_score(y_test, y_pred)
    except ValueError:
        # Handle case where only one class is present in y_true
        auc = 0.5
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average="binary")
    precision = metrics.precision_score(y_test, y_pred, average="binary")
    recall = metrics.recall_score(y_test, y_pred, average="binary")
    accuracy = metrics.accuracy_score(y_test, y_pred)

    if verbose >= 1:
        logger.info("Ensemble MCC %s, AUC %s, nb %s", mcc, auc, len(individual[0]))

    # ?? how does the diversity weighting and the order of operations interact with the ensemble weights already
    # measure and incorporate diversity
    diversity_metric = measure_diversity_wrapper(individual, method="comprehensive")

    diversity_parameter = local_param_dict.get(
        "div_p"
    )  # user specified, if >0, div score used.

    diversity_params = {
        "penalty_method": "linear",  # or "quadratic", "exponential", "threshold"
        "penalty_strength": local_param_dict.get(
            "div_p", 0.3
        ),  # User specified penalty magnitude.
        "min_score_factor": 0.1,  # Prevent scores going below 10%
        "similarity_threshold": 0.7,  # For threshold method
    }

    auc_div, mcc_div = apply_diversity_penalty(
        auc, mcc, diversity_metric, diversity_params
    )

    # blank init with headers in main log_store_dataframe_path
    ensemble_model_list = []
    feature_count_list = []
    auc_score_list = []
    mcc_score_list = []
    # For each member in the ensemble
    for i in range(0, len(individual[0]) - 1):
        ensemble_model_list.append(str(individual[0][i][1]))  # str or model?
        feature_count_list.append(
            individual[0][i][2]
        )  # could replace with length of list to reduce size on disk
        auc_score_list.append(individual[0][i][4])
        mcc_score_list.append(individual[0][i][0])

    # orignal_feature_names

    feature_map_vector = []
    for col in original_feature_names:
        if col in feature_count_list:
            feature_map_vector.append(1)
        else:
            feature_map_vector.append(0)

    feature_map_vector = np.array(feature_map_vector)

    # set score log dataframe in main
    df_data = [
        [
            len(individual[0]),
            auc,
            np.mean(auc_score_list),
            auc_div,
            mcc,
            np.mean(mcc_score_list),
            mcc_div,
            f1,
            precision,
            recall,
            accuracy,
            ensemble_model_list,
            feature_map_vector,
            auc_score_list,
            mcc_score_list,
        ]
    ]
    column_headers = [
        "n",
        "auc",
        "auc_mean",
        "auc_div",
        "mcc",
        "mcc_mean",
        "mcc_div",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "ensemble_model_list",
        "feature_count_list",
        "auc_score_list",
        "mcc_score_list",
    ]

    #     column_headers = ['nb_size', 'f_list', 'auc','mcc','f1','precision','recall','accuracy', 'nb_val', 'pop_val', 'g_val', 'g', 'weighted', 'use_stored_base_learners', 'store_base_learners',
    #        'resample', 'scale', 'n_features', 'param_space_size', 'n_unique_out',
    #        'outcome_var_n', 'div_p', 'percent_missing', 'corr',
    #                        'age', 'sex', 'bmi','ethnicity', 'bloods', 'diagnostic_order',
    #                       'drug_order', 'annotation_n', 'meta_sp_annotation_n',
    #                       'X_train_size', 'X_test_orig_size', 'X_test_size',
    #                    'run_time', 'cxpb', 'mutpb', 'indpb', 't_size']

    df = pd.DataFrame(data=df_data, columns=column_headers)
    df.to_csv(
        f"{file_path}/progress_logs_scores/{log_store_dataframe_path}.csv",
        mode="a",
        index=True,
        header=False,
    )

    if verbose >= 1:
        logger.info(
            f"""Ensemble MCC {mcc}, diversity weighted MCC {mcc_div}
        , \n f1 {f1} precision {precision} recall {recall} accuracy {accuracy}
        , \n AUC {auc}, diversity weighted AUC {auc_div}
        , \n nb {len(individual[0])}, diversity_score: {diversity_metric}, diff: {auc_div-auc} """
        )

    # return mcc for genetic algorithm evalutation
    if diversity_parameter > 0:
        return (auc_div,)
    else:
        return (auc,)
    # return (auc,)


round_v = np.vectorize(round)


def normalize(weights: np.ndarray) -> np.ndarray:
    """Normalizes a vector of weights using the L1 norm.

    If the input vector is all zeros, it is returned as is. Otherwise, each
    element is divided by the L1 norm (sum of absolute values) so that the
    sum of the absolute values of the new vector is 1.

    Args:
        weights: The NumPy array of weights to normalize.

    Returns:
        The L1-normalized weight vector.
    """
    result = norm(weights, 1)
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result


def measure_binary_vector_diversity(ensemble: List, metric: str = "jaccard") -> float:
    """Calculates the diversity of an ensemble based on their prediction vectors.

    This function uses `scipy.spatial.distance.pdist` to calculate the pairwise
    distance between the prediction vectors of the base learners in an ensemble.
    The mean of these distances is returned as the diversity score.

    Args:
        ensemble: The ensemble to evaluate, structured as a list of lists of
            model tuples.
        metric: The distance metric to use (e.g., 'jaccard', 'hamming').
            Defaults to "jaccard".

    Returns:
        The mean distance between prediction vectors, representing the
        ensemble's diversity.
    """
    n_y_pred = len(ensemble[0])  # check level

    all_y_pred_arrays = []

    for i in range(0, n_y_pred):
        all_y_pred_arrays.append(ensemble[0][i][5])

    distance_vector = scipy.spatial.distance.pdist(all_y_pred_arrays, metric=metric)

    return np.mean(distance_vector)
