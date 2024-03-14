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
                        0.0, min(float(subsample[i]), 1.0)
                    )
                    print(
                        f"Invalid value for subsample[{i}]. Setting it to a value within the valid range."
                    )
        elif not isinstance(subsample, float) or subsample <= 0.0 or subsample > 1.0:
            # If subsample is not a list but a single value, validate it directly
            param_space["subsample"] = max(0.0, min(float(subsample), 1.0))
            print(
                "Invalid value for subsample. Setting it to a value within the valid range."
            )
    return param_space
