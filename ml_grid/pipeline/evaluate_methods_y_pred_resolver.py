from typing import Any, List, Union
import numpy as np
from ml_grid.ga_functions.ga_eval_ann_weight_method import (
    get_y_pred_ann_torch_weighting_eval,
)
from ml_grid.ga_functions.ga_eval_de_weight_method import (
    get_weighted_ensemble_prediction_de_y_pred_valid_eval,
)
from ml_grid.ga_functions.ga_eval_ensemble_weight_finder_de import (
    super_ensemble_weight_finder_differential_evolution_eval,
)
from ml_grid.ga_functions.ga_eval_unweighted import get_best_y_pred_unweighted_eval
from numpy.linalg import norm


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
        print("get_y_pred_resolver")
        print(ensemble)
    local_param_dict = ml_grid_object.local_param_dict
    X_test_orig = ml_grid_object.X_test_orig
    y_test_orig = ml_grid_object.y_test_orig

    if ml_grid_object.verbose >= 2:
        print("Starting get_y_pred_resolver function...")
        print("local_param_dict:", local_param_dict)
        print("X_test_orig shape:", X_test_orig.shape)
        print("y_test_orig shape:", y_test_orig.shape)

    if (
        local_param_dict.get("weighted") == None
        or local_param_dict.get("weighted") == "unweighted"
    ):
        if ml_grid_object.verbose >= 1:
            print("Using unweighted ensemble prediction...")
        try:
            y_pred = get_best_y_pred_unweighted_eval(
                ensemble, ml_grid_object, valid=valid
            )
        except Exception as e:
            print(
                "exception on y_pred = get_best_y_pred_unweighted(ensemble, ml_grid_object, valid=valid)"
            )
            print(ensemble)
            print("valid", valid)
            raise e
    elif local_param_dict.get("weighted") == "de":
        if ml_grid_object.verbose >= 1:
            print("Using DE weighted ensemble prediction...")
        y_pred = get_weighted_ensemble_prediction_de_y_pred_valid_eval(
            ensemble,
            super_ensemble_weight_finder_differential_evolution_eval(
                ensemble, ml_grid_object, valid=valid
            ),
            ml_grid_object,
            valid=valid,
        )
        if ml_grid_object.verbose >= 2:
            print("DE weighted y_pred shape:", y_pred.shape)
    elif local_param_dict.get("weighted") == "ann":
        if ml_grid_object.verbose >= 1:
            print("Using ANN weighted ensemble prediction...")
        y_pred = get_y_pred_ann_torch_weighting_eval(
            ensemble, ml_grid_object, valid=valid
        )

    return y_pred
