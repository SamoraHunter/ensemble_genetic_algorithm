import logging
from typing import Any

import pandas as pd

logger = logging.getLogger("ensemble_ga")


def debug_base_learner(
    model: Any,
    mccscore: float,
    X_train: pd.DataFrame,
    auc_score: float,
    model_train_time: int,
) -> None:
    """Prints debug information for a single trained base learner.

    This function is called within model generator functions when the verbosity
    level is high enough. It prints a summary of the model's performance,
    including its type, MCC score, number of features used, AUC score, and
    training time. It also issues a warning if the training time exceeds a
    predefined threshold.

    Args:
        model: The trained model object.
        mccscore: The Matthews Correlation Coefficient of the model.
        X_train: The training data DataFrame, used to get the feature count.
        auc_score: The ROC AUC score of the model.
        model_train_time: The time taken to train the model, in seconds.
    """
    from ml_grid.util.global_params import global_parameters

    global_parameters_vals = global_parameters()

    model_train_time_warning_threshold = (
        global_parameters_vals.model_train_time_warning_threshold
    )

    logger.debug(
        "%s, %s, %s, %s, %s",
        str(model).split("(")[0],
        round(mccscore, 5),
        len(X_train.columns),
        auc_score,
        model_train_time,
    )
    if model_train_time > model_train_time_warning_threshold:
        logger.warning("Warning long train time, ")
