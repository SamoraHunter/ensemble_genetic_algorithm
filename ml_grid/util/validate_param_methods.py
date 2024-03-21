import numpy as np


def validate_max_leaf_nodes(param_space):
    if "max_leaf_nodes" in param_space:
        max_leaf_nodes = param_space["max_leaf_nodes"]
        if not isinstance(max_leaf_nodes, int) or max_leaf_nodes < 2:
            # Set a default value if max_leaf_nodes is invalid
            param_space["max_leaf_nodes"] = 2  # Or any other valid value you prefer
            print("Invalid value for max_leaf_nodes. Setting it to default value.")
    return param_space


def validate_batch_size(param_space):
    if "batch_size" in param_space:
        batch_size = param_space["batch_size"]
        if not isinstance(batch_size, int) or batch_size < 2:
            # Set a default value if batch_size is invalid
            param_space["batch_size"] = 2  # Or any other valid value you prefer
            print("Invalid value for batch_size. Setting it to default value.")
    return param_space


def validate_subsample(param_space):
    try:
        if "subsample" in param_space:
            subsample = param_space["subsample"]
            if isinstance(subsample, list):
                # If subsample is a list, iterate over its elements and validate each one
                for i in range(len(subsample)):
                    if (
                        not isinstance(subsample[i], float)
                        or subsample[i] <= 0.0
                        or subsample[i] > 1.0
                    ):
                        # Set a default value within the valid range if subsample is invalid
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
                    # If subsample is not a list but a single value, validate it directly
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


def validate_warm_start(param_space):
    if "warm_start" in param_space:
        warm_start = param_space["warm_start"]
        if not isinstance(warm_start, bool) and not isinstance(warm_start, np.bool_):
            # Set a default value if warm_start is invalid
            param_space["warm_start"] = True  # Or any other valid value you prefer
            print("Invalid value for warm_start. Setting it to default value.")
    return param_space


def validate_min_samples_split(param_space):
    if "min_samples_split" in param_space:
        min_samples_split = param_space["min_samples_split"]
        if not (isinstance(min_samples_split, int) and min_samples_split >= 2) and not (
            isinstance(min_samples_split, float) and 0.0 < min_samples_split < 1.0
        ):
            # Set a default value if min_samples_split is invalid
            param_space["min_samples_split"] = 2  # Or any other valid value you prefer
            print("Invalid value for min_samples_split. Setting it to default value.")
    return param_space
