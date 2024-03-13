def validate_max_leaf_nodes(param_space):
    if "max_leaf_nodes" in param_space:
        max_leaf_nodes = param_space["max_leaf_nodes"]
        if not isinstance(max_leaf_nodes, int) or max_leaf_nodes < 2:
            # Set a default value if max_leaf_nodes is invalid
            param_space["max_leaf_nodes"] = 2  # Or any other valid value you prefer
            print("Invalid value for max_leaf_nodes. Setting it to default value.")
    return param_space
