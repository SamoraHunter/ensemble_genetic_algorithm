import os
import yaml
import logging
from typing import Dict, Any

# A flag to ensure the config-related message is printed only once per run.
_config_message_printed = False

# Use a named logger for this module
logger = logging.getLogger("ensemble_ga")
logger.setLevel(logging.INFO)


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
    global _config_message_printed
    if os.path.exists(config_path):
        if not _config_message_printed:
            logger.info(f"Loading custom configuration from '{config_path}'")
            _config_message_printed = True
        with open(config_path, "r") as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                logger.error(f"Could not parse YAML file '{config_path}': {e}")
                return {}
    else:
        if not _config_message_printed:
            # This message can be spammy, so we keep it concise.
            logger.info("No 'config.yml' found, using default parameters.")
            _config_message_printed = True
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
            # Ensure we only merge into existing keys to prevent infinite recursion
            user_data_dict = value[0]
            default_data_dict = default[key][0]
            valid_user_data = {k: v for k, v in user_data_dict.items() if k in default_data_dict}
            default[key][0] = merge_configs(default_data_dict, valid_user_data)
        else:
            default[key] = value
    return default