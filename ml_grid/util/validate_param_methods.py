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
