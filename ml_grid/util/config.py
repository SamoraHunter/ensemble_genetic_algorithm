import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    """
    Loads a YAML configuration file from the given path.

    This function searches for the configuration file and, if found, loads it.
    It provides a centralized way to manage user-defined settings.

    Args:
        config_path: The path to the YAML configuration file.
                     Defaults to "config.yml" in the current directory.

    Returns:
        A dictionary containing the configuration, or an empty dictionary
        if the file is not found.
    """
    if os.path.exists(config_path):
        print(f"INFO: Loading custom configuration from '{config_path}'")
        with open(config_path, "r") as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                print(f"ERROR: Could not parse YAML file '{config_path}': {e}")
                return {}
    else:
        print("INFO: No 'config.yml' found. Using default parameters.")
        return {}


def merge_configs(default: Dict, user: Dict) -> Dict:
    """
    Recursively merges a user-defined configuration into the default one.

    Args:
        default: The default configuration dictionary.
        user: The user's configuration dictionary to merge.

    Returns:
        The merged configuration dictionary.
    """
    for key, value in user.items():
        if key in default and isinstance(default[key], dict) and isinstance(value, dict):
            default[key] = merge_configs(default[key], value)
        # Special handling for 'data' key, which contains a list of dictionaries
        elif (
            key == "data"
            and key in default
            and isinstance(default[key], list)
            and isinstance(value, list)
            and default[key] and value and isinstance(default[key][0], dict) and isinstance(value[0], dict)
        ):
            default[key][0] = merge_configs(default[key][0], value[0])
        else:
            default[key] = value
    return default