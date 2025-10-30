from typing import Any, List, Union
import logging
import numpy as np
from ml_grid.ga_functions.ga_eval_ann_weight_method import (
    get_ann_weighted_ensemble_predictions_eval,
)
from ml_grid.ga_functions.ga_eval_de_weight_method import ( # Renamed
    get_de_weighted_ensemble_predictions_eval,
)
from ml_grid.ga_functions.ga_eval_ensemble_weight_finder_de import (
    find_ensemble_weights_de_eval,
)
from ml_grid.ga_functions.ga_eval_unweighted import get_unweighted_ensemble_predictions_eval
from numpy.linalg import norm
logger = logging.getLogger("ensemble_ga")


def get_y_pred_resolver_eval(
    ensemble: List, ml_grid_object: Any, valid: bool = False
) -> Union[List, np.ndarray]:
    """Resolves and generates predictions for an ensemble during final evaluation.

    This function acts as a dispatcher, calling the appropriate evaluation-specific
    prediction generation function (`unweighted`, `de` for Differential Evolution,
    or `ann` for an Artificial Neural Network) based on the 'weighted' parameter in
    the `local_param_dict`.

    Unlike its training-time counterpart (`get_y_pred_resolver`), this function
    calls evaluation-specific methods that always refit the models on the full
    training data before making predictions on the test or validation set.

    Args:
        ensemble: A list containing the ensemble configuration.
        ml_grid_object: The main experiment object, containing data splits and
            configuration parameters.
        valid: If True, predictions are generated for the validation set.
            If False, predictions are for the test set. Defaults to False.

    Returns:
        The final ensemble predictions, as either a list or a NumPy array.
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
        local_param_dict.get("weighted") == None
        or local_param_dict.get("weighted") == "unweighted"
    ):
        if ml_grid_object.verbose >= 1:
            logger.info("Using unweighted ensemble prediction...")
        try:
            y_pred = get_unweighted_ensemble_predictions_eval(
                ensemble, ml_grid_object, valid=valid
            )
        except Exception as e:
            logger.error(
                "exception on y_pred = get_best_y_pred_unweighted(ensemble, ml_grid_object, valid=valid)"
            )
            logger.error(ensemble)
            logger.error("valid: %s", valid)
            raise e
    elif local_param_dict.get("weighted") == "de":
        if ml_grid_object.verbose >= 1:
            logger.info("Using DE weighted ensemble prediction...")
        y_pred = get_de_weighted_ensemble_predictions_eval(
            ensemble,
            find_ensemble_weights_de_eval(ensemble, ml_grid_object, valid=valid),
            ml_grid_object,
            valid=valid,
        )
        if ml_grid_object.verbose >= 2:
            logger.info("DE weighted y_pred shape: %s", y_pred.shape)
    elif local_param_dict.get("weighted") == "ann":
        if ml_grid_object.verbose >= 1:
            logger.info("Using ANN weighted ensemble prediction...")
        y_pred = get_ann_weighted_ensemble_predictions_eval(
            ensemble, ml_grid_object, valid=valid
        )

    return y_pred
