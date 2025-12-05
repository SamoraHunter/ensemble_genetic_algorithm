import logging
import os
import pickle
from typing import Dict, List

logger = logging.getLogger("ensemble_ga")


def handle_percent_missing(
    local_param_dict: Dict, all_df_columns: List[str], drop_list: List[str]
) -> List[str]:
    """Removes columns with a high percentage of missing data.

    This function identifies columns to be dropped based on a pre-calculated
    dictionary of missing data percentages. It expects a file named
    'percent_missing_dict.pickle' in the root directory. It compares the
    missing percentage of each column from this file against a threshold
    specified in `local_param_dict`.

    Args:
        local_param_dict: A dictionary of parameters for the current pipeline,
            expected to contain the 'percent_missing' threshold.
        all_df_columns: A list of all column names in the dataframe to be processed.
        drop_list: The list of columns already marked for dropping. This list
            will be extended with columns that exceed the missing data threshold.

    Returns:
        The updated list of columns to be dropped from the dataframe.
    """
    # Check for null pointer references
    assert local_param_dict is not None
    assert all_df_columns is not None
    assert drop_list is not None

    percent_missing_drop_list = []

    # Check if the file exists
    if os.path.exists("percent_missing_dict.pickle"):
        with open("percent_missing_dict.pickle", "rb") as handle:
            try:
                percent_missing_dict = pickle.load(handle)
            except (pickle.UnpicklingError, EOFError) as e:
                logger.warning("Error loading pickle file: %s. Treating as empty.", e)
                percent_missing_dict = {}
    else:
        logger.info(
            "File 'percent_missing_dict.pickle' not found. Skipping missing data check."
        )
        percent_missing_dict = {}

    percent_missing_threshold = local_param_dict.get("percent_missing")
    if percent_missing_threshold is not None and percent_missing_dict:

        # Iterate through columns
        for col in all_df_columns:
            # Try to get the value from the dictionary
            try:
                if (
                    col in percent_missing_dict
                    and percent_missing_dict.get(col) > percent_missing_threshold
                ):
                    percent_missing_drop_list.append(col)
            except (TypeError, ValueError) as e:
                # This can happen if a value in percent_missing_dict is not a number
                logger.warning(
                    "Warning: Could not compare missing percentage for column '%s'. Value was not a number. Error: %s",
                    col,
                    e,
                )

        logger.info(
            f"Identified {len(percent_missing_drop_list)} columns with > {percent_missing_threshold} percent missing data."
        )

        # Extend the drop list with identified columns
        drop_list.extend(percent_missing_drop_list)

    else:
        logger.info(
            "percent_missing_threshold is None or percent_missing_dict is empty. Skipping percent missing data check."
        )

    return drop_list
