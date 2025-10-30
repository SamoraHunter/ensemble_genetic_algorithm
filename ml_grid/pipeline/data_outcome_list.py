import logging
from typing import List

from ml_grid.util import outcome_list

logger = logging.getLogger("ensemble_ga")


def handle_outcome_list(drop_list: List[str], outcome_variable: str) -> List[str]:
    """Ensures all potential outcome variables are dropped except the target one.

    This function takes a list of columns to be dropped, extends it with a
    predefined list of all possible outcome variables from `outcome_list`,
    and then specifically removes the currently targeted `outcome_variable`
    from the drop list. This is a safety measure to prevent any potential
    target leakage from other outcome columns.

    Args:
        drop_list: The list of columns already marked for dropping.
        outcome_variable: The name of the target outcome variable, which
            should NOT be dropped.

    Returns:
        The updated list of columns to drop.
    """
    logger.info("Extending all outcome list on drop list")

    outcome_object = outcome_list.OutcomeList()

    outcome_list_list: List[str] = outcome_object.all_outcome_list

    drop_list.extend(outcome_list_list)

    try:
        drop_list.remove(outcome_variable)
    except ValueError:
        logger.warning("Warning: Target outcome '%s' was not in the master outcome list to begin with.", outcome_variable)

    return drop_list
