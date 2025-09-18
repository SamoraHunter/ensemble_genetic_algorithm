import numpy as np
from typing import Any, Dict


def validate_max_leaf_nodes(param_space: Dict[str, Any]) -> Dict[str, Any]:
    """Validates the 'max_leaf_nodes' parameter in a hyperparameter space.

    Ensures that `max_leaf_nodes`, if present, is an integer greater than or
    equal to 2. If not, it is set to a default value of 2.

    Args:
        param_space: The dictionary of hyperparameters.

    Returns:
        The (potentially modified) dictionary of hyperparameters.
    """
    if "max_leaf_nodes" in param_space:
        max_leaf_nodes = param_space["max_leaf_nodes"]
        if not isinstance(max_leaf_nodes, int) or max_leaf_nodes < 2:
            param_space["max_leaf_nodes"] = 2  # Or any other valid value you prefer
            print("Invalid value for max_leaf_nodes. Setting it to default value.")
    return param_space


def hidden_layer_size(param_space):
    if "hidden_layer_size" in param_space:
        hidden_size = param_space["hidden_layer_size"]
        if not isinstance(hidden_size, int) or hidden_size < 2:
            param_space["hidden_layer_size"] = 2  # Or any other valid value you prefer
            print("Invalid value for hidden_layer_size. Setting it to default value.")
    return param_space


def validate_subsample(param_space: Dict[str, Any]) -> Dict[str, Any]:
    """Validates the 'subsample' parameter in a hyperparameter space.

    Ensures that `subsample`, if present, is a float between 0.0 and 1.0.
    If it is outside this range, it is clamped to the valid range (with a
    minimum of 0.01). This function handles both single float values and
    lists of floats.

    Args:
        param_space: The dictionary of hyperparameters.

    Returns:
        The (potentially modified) dictionary of hyperparameters.
    """
    try:
        if "subsample" in param_space:
            subsample = param_space["subsample"]
            if isinstance(subsample, list):
                for i in range(len(subsample)):
                    if (
                        not isinstance(subsample[i], float)
                        or subsample[i] <= 0.0
                        or subsample[i] > 1.0
                    ):
                        param_space["subsample"][i] = max(
                            0.01, min(float(subsample[i]), 1.0)
                        )  # Change default value to 0.01
                        print(
                            f"Invalid value for subsample[{i}]. Setting it to a value within the valid range."
                        )
            else:
                if (
                    not isinstance(subsample, float)
                    or subsample <= 0.0
                    or subsample > 1.0
                ):
                    param_space["subsample"] = max(
                        0.01, min(float(subsample), 1.0)
                    )  # Change default value to 0.01
                    print(
                        "Invalid value for subsample. Setting it to a value within the valid range."
                    )
    except Exception as e:
        print("Error occurred. Input param_space:", param_space)
        raise e
    return param_space


def validate_warm_start(param_space: Dict[str, Any]) -> Dict[str, Any]:
    """Validates the 'warm_start' parameter in a hyperparameter space.

    Ensures that `warm_start`, if present, is a boolean value. If not, it is
    set to a default value of True.

    Args:
        param_space: The dictionary of hyperparameters.

    Returns:
        The (potentially modified) dictionary of hyperparameters.
    """
    if "warm_start" in param_space:
        warm_start = param_space["warm_start"]
        if not isinstance(warm_start, bool) and not isinstance(warm_start, np.bool_):
            param_space["warm_start"] = True  # Or any other valid value you prefer
            print("Invalid value for warm_start. Setting it to default value.")
    return param_space


def validate_min_samples_split(param_space: Dict[str, Any]) -> Dict[str, Any]:
    """Validates the 'min_samples_split' parameter in a hyperparameter space.

    Ensures that `min_samples_split`, if present, is either an integer >= 2
    or a float between 0.0 and 1.0. If not, it is set to a default value of 2.

    Args:
        param_space: The dictionary of hyperparameters.

    Returns:
        The (potentially modified) dictionary of hyperparameters.
    """
    if "min_samples_split" in param_space:
        min_samples_split = param_space["min_samples_split"]
        if not (isinstance(min_samples_split, int) and min_samples_split >= 2) and not (
            isinstance(min_samples_split, float) and 0.0 < min_samples_split < 1.0
        ):
            param_space["min_samples_split"] = 2  # Or any other valid value you prefer
            print("Invalid value for min_samples_split. Setting it to default value.")
    return param_space
