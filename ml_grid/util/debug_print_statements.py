
from numpy import absolute, mean, std
from sklearn import metrics
from sklearn.metrics import (classification_report, f1_score, make_scorer,
                             matthews_corrcoef, roc_auc_score) # Removed unused imports
from typing import Dict, Any


class debug_print_statements_class():
    """A utility class for printing formatted debug information about model scores.

    This class provides a static method to display mean and standard deviation
    of various cross-validation scores, such as F1, ROC AUC, accuracy, fit time,
    and score time.
    """

    scores: Dict[str, Any]
    """A dictionary containing cross-validation scores, typically from `sklearn.model_selection.cross_validate`."""

    def __init__(self, scores: Dict[str, Any]):
        """Initializes the debug_print_statements_class.

        Args:
            scores: A dictionary containing cross-validation scores. Expected keys
                include 'test_f1', 'test_roc_auc', 'test_accuracy', 'fit_time',
                and 'score_time'.
        """
        self.scores = scores

    @staticmethod
    def debug_print_scores(scores: Dict[str, Any]) -> None:
        """Prints formatted mean and standard deviation of various scores.

        Args:
            scores: A dictionary containing cross-validation scores. Expected keys
                include 'test_f1', 'test_roc_auc', 'test_accuracy', 'fit_time',
                and 'score_time'.
        """
        print(
            "Mean MAE: %.3f (%.3f)"
            % (
                absolute(mean(scores["test_f1"])),
                std(scores["test_f1"]),
            )
        )
        print(
            "Mean ROC AUC: %.3f (%.3f)"
            % (
                absolute(mean(scores["test_roc_auc"])),
                std(scores["test_roc_auc"]),
            )
        )
        print(
            "Mean accuracy: %.3f (%.3f)"
            % (absolute(mean(scores["test_accuracy"])), std(scores["test_accuracy"]))
        )
        print(
            "Mean fit time: %.3f (%.3f)"
            % (absolute(mean(scores["fit_time"])), std(scores["fit_time"]))
        )
        print(
            "Mean score time: %.3f (%.3f)"
            % (absolute(mean(scores["score_time"])), std(scores["score_time"]))
        )
        print(
            "---------------------------------------------------------------------------------------------------"
        )