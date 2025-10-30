# A module for evaluating ensembles by calling the prediction resolver function.

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from typing import Any, Dict, List, Optional, Tuple


from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from ml_grid.ga_functions.ga_ann_util import (
    BinaryClassification,
    TestData,
    TrainData,
    binary_acc,
    get_free_gpu,
)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import ElasticNet
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from ml_grid.model_classes_ga.perceptron_dummy_model import perceptronModelGen_dummy
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from numpy import array

logger = logging.getLogger("ensemble_ga")
try:
    from ml_grid.pipeline.data_train_test_split import get_data_split
except ImportError as e:
    logger.error("Error: Could not import required functions from 'ml_grid': %s", e)

from ml_grid.pipeline.evaluate_methods_y_pred_resolver import (
                            get_y_pred_resolver_eval,
                        )


# --- Helper Class to Mimic MLGridObject ---
class _MLGridObject:
    """A minimal container class to mimic the main MLGridObject for evaluation.

    This class holds the necessary data splits and configuration parameters
    required by the `get_y_pred_resolver_eval` function.
    """

    verbose: int
    """The verbosity level for logging."""
    local_param_dict: Dict
    """A dictionary of local parameters, including the weighting method."""
    X_train: pd.DataFrame
    """The training features DataFrame."""
    y_train: pd.Series
    """The training target Series."""
    X_test: pd.DataFrame
    """The testing features DataFrame."""
    y_test: pd.Series
    """The testing target Series."""
    X_test_orig: pd.DataFrame
    """The original (validation) testing features DataFrame."""
    y_test_orig: pd.Series
    """The original (validation) testing target Series."""

    def __init__(self):
        """Initializes the _MLGridObject."""
        self.verbose = 0
        self.local_param_dict: Dict[str, Any] = {}
        self.X_train, self.y_train = pd.DataFrame(), pd.Series()
        self.X_test, self.y_test = pd.DataFrame(), pd.Series()
        self.X_test_orig, self.y_test_orig = pd.DataFrame(), pd.Series()


class EnsembleEvaluator:
    """Evaluates stored ensembles from a results CSV file.

    This class loads a results DataFrame, reconstructs the ensembles, and
    re-evaluates them on specified data splits (test or validation) using
    different weighting methods. It is a powerful tool for post-hoc analysis
    of genetic algorithm runs.
    """

    debug: bool
    """Flag to enable detailed debug printing."""

    ml_grid_object: _MLGridObject
    """A container holding the data splits and configuration for evaluation."""

    original_feature_names: Optional[List[str]]
    """The complete list of original feature names from the dataset."""

    def __init__(
        self,
        input_csv_path: str,
        outcome_variable: str,
        initial_param_dict: dict,
        debug: bool = False,
    ):
        """Initializes the EnsembleEvaluator.

        Args:
            input_csv_path: The path to the input CSV file containing the raw data.
            outcome_variable: The name of the target variable column in the CSV.
            initial_param_dict: A dictionary of parameters required for data
                splitting (e.g., `{'resample': None}`).
            debug: If True, enables verbose debug printing. Defaults to False.
        """
        self.debug = debug
        if self.debug:
            logger.debug("--- Initializing EnsembleEvaluator ---")
        self.ml_grid_object = _MLGridObject()
        self.ml_grid_object.local_param_dict = initial_param_dict
        self.ml_grid_object.verbose = 0
        self.original_feature_names = None
        self._load_and_split_data(input_csv_path, outcome_variable)

    def _load_and_split_data(
        self, input_csv_path: str, outcome_variable: str
    ) -> None:
        """Loads data from a CSV and splits it into train, test, and validation sets."""
        if self.debug:
            logger.debug("Loading data from: %s", input_csv_path)
        df = pd.read_csv(input_csv_path)
        y = df[outcome_variable]
        X = df.drop(outcome_variable, axis=1)
        self.original_feature_names = list(X.columns)
        if self.debug:
            logger.debug("Splitting data and assigning to ml_grid_object...")
        X_train, X_test, y_train, y_test, X_val, y_val = get_data_split(
            X, y, self.ml_grid_object.local_param_dict
        )
        self.ml_grid_object.X_train, self.ml_grid_object.y_train = X_train, y_train
        self.ml_grid_object.X_test, self.ml_grid_object.y_test = X_test, y_test
        self.ml_grid_object.X_test_orig, self.ml_grid_object.y_test_orig = X_val, y_val
        if self.debug:
            logger.debug("Data splits assigned successfully.")

    def _parse_ensemble(
        self, ensemble_record: Any, debug: bool = False
    ) -> List[List[Tuple]]:
        """Parses a 'best_ensemble' record into a list of processed ensembles.

        This method takes a string or list representation of an ensemble from the
        results DataFrame and reconstructs the model objects using `eval()`.

        Args:
            ensemble_record: The raw 'best_ensemble' entry from a DataFrame row.
            debug: If True, enables verbose debug printing for the parsing process.

        Returns:
            A list of ensembles, where each ensemble is a list of model tuples.

        Warning:
            This function uses `eval()` to reconstruct model objects from
            their string representation. This can be a security risk if the
            input data is from an untrusted source.
        """
        import pandas as pd
        from numpy import array

        # If it's a pandas Series, extract the first value
        if isinstance(ensemble_record, pd.Series):
            if debug:
                logger.debug(
                    f"[DEBUG] ensemble_record is a Series, extracting first value: {ensemble_record}"
                )
            ensemble_record = ensemble_record.iloc[0]
        # If it's a string, eval it; if not, assume it's already a list/tuple
        if isinstance(ensemble_record, str):
            try:
                ensembles = eval(ensemble_record, {"array": array})
            except Exception as e:
                if debug:
                    logger.debug("Could not eval ensemble_record: %s", e)
                return []
        else:
            ensembles = ensemble_record
        if debug:
            logger.debug("[DEBUG] ensembles after eval: %s", ensembles)
        processed_ensembles = []
        # If ensembles is a Series, extract first value
        if hasattr(ensembles, "iloc") and not isinstance(ensembles, (list, tuple)):
            if debug:
                logger.debug(
                    f"[DEBUG] ensembles is a Series-like, extracting first value: {ensembles}"
                )
            ensembles = ensembles.iloc[0]
        # If it's a single ensemble, wrap in a list
        if (
            isinstance(ensembles, (list, tuple))
            and len(ensembles) > 0
            and not isinstance(ensembles[0], (list, tuple))
        ):
            ensembles = [ensembles]
        for ensemble in ensembles:
            if not isinstance(ensemble, (list, tuple)):
                if debug:
                    logger.debug("[DEBUG] Skipping non-list/tuple ensemble: %s", ensemble)
                continue
            processed_ensemble = []
            for model_tuple in ensemble:
                if not isinstance(model_tuple, (list, tuple)) or len(model_tuple) < 2:
                    if debug:
                        logger.debug(
                            f"Skipping invalid model_tuple (expected at least 2 elements): {model_tuple}"
                        )
                    continue
                model_string = model_tuple[1]
                try:
                    model_object = eval(model_string)
                    new_tuple = list(model_tuple)
                    new_tuple[1] = model_object
                    processed_ensemble.append(tuple(new_tuple))
                except Exception as e:
                    if debug:
                        logger.debug(
                            f"Could not eval model string: '{model_string}'. Error: {e}"
                        )
                    continue
            processed_ensembles.append(processed_ensemble)
        return processed_ensembles

    def _run_evaluation_from_df(
        self,
        results_df: pd.DataFrame,
        weighting_methods: List[str],
        use_validation_set: bool = False,
        ensemble_indices: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Core evaluation loop for a given set of weighting methods.

        Args:
            results_df: The DataFrame containing the GA run results.
            weighting_methods: A list of weighting methods to evaluate
                (e.g., ['unweighted', 'de', 'ann']).
            use_validation_set: If True, evaluates on the validation set;
                otherwise, evaluates on the test set.
            ensemble_indices: A list of row indices from `results_df` to evaluate.
                If None, evaluates the single best run by 'auc_score'.
        Returns:
            A DataFrame containing the performance metrics for each evaluated
            ensemble and weighting method.
        """
        import pandas as pd


        if ensemble_indices is None:
            # Default: pick the row with the highest auc_score
            if "auc_score" in results_df.columns:
                idx = results_df["auc_score"].idxmax()
            else:
                idx = 0
            selected_rows = [idx]
        elif ensemble_indices == "all":
            selected_rows = list(results_df.index)
        elif isinstance(ensemble_indices, int):
            selected_rows = [ensemble_indices]
        else:
            selected_rows = list(ensemble_indices)

        all_results = []
        for row_idx in selected_rows:
            row = results_df.loc[row_idx]

            # Ensure we only pass a single value, not a Series
            ensemble_record = row["best_ensemble"]
            if hasattr(ensemble_record, "iloc") and not isinstance(
                ensemble_record, (list, tuple, str)
            ):
                logger.debug(
                    f"[DEBUG] ensemble_record is a Series-like in _run_evaluation_from_df, extracting first value: {ensemble_record}"
                )
                ensemble_record = ensemble_record.iloc[0]

            processed_ensembles = self._parse_ensemble(
                ensemble_record, debug=self.debug
            )

            # Convert mask to feature names for each ensemble
            def mask_to_features(mask, feature_names):
                # mask can be a list/array of 0/1 or bool, or indices
                if isinstance(mask, (list, tuple, np.ndarray)):
                    # If mask is bool or 0/1 and same length as features
                    if len(mask) == len(feature_names) and all(
                        isinstance(x, (int, np.integer, bool, np.bool_)) for x in mask
                    ):
                        return [
                            fname
                            for fname, m in zip(feature_names, mask)
                            if int(m) == 1
                        ]
                    # If mask is indices
                    elif all(
                        isinstance(x, (int, np.integer)) and 0 <= x < len(feature_names)
                        for x in mask
                    ):
                        return [feature_names[x] for x in mask]
                return mask  # fallback, return as is

            for i, ensemble in enumerate(processed_ensembles):
                # For each model_tuple in ensemble, convert only the mask (3rd element) to feature names, keep all other elements unchanged
                ensemble_with_features = []
                for model_tuple in ensemble:
                    if len(model_tuple) >= 3:
                        mask = model_tuple[2]
                        feature_names = mask_to_features(
                            mask, self.original_feature_names
                        )
                        new_tuple = list(model_tuple)
                        new_tuple[2] = feature_names
                        ensemble_with_features.append(tuple(new_tuple))
                    else:
                        ensemble_with_features.append(model_tuple)

                for weight_method in weighting_methods:
                    # Set the weighting method in local_param_dict
                    self.ml_grid_object.local_param_dict["weighted"] = weight_method

                    try:
                        
                        # Call the resolver function directly - it handles all masking internally
                        y_pred = get_y_pred_resolver_eval(
                            [ensemble_with_features],
                            self.ml_grid_object,
                            valid=use_validation_set,
                        )

                        # Get true labels
                        if use_validation_set:
                            y_true = self.ml_grid_object.y_test_orig
                        else:
                            y_true = self.ml_grid_object.y_test

                        # Debug: print lengths and feature names
                        if self.debug:
                            logger.debug(
                                f"[DEBUG] y_true length: {len(y_true)}, y_pred length: {len(y_pred)}"
                            )
                            logger.debug(
                                f"[DEBUG] Features used in ensemble: {[t[2] for t in ensemble_with_features]}"
                            )

                        # Calculate metrics
                        result = {
                            "row_index": row_idx,
                            "ensemble_id": i + 1,
                            "weighting_method": weight_method,
                            "accuracy": accuracy_score(y_true, y_pred),
                            "precision": precision_score(
                                y_true, y_pred, zero_division=0
                            ),
                            "recall": recall_score(y_true, y_pred, zero_division=0),
                            "f1_score": f1_score(y_true, y_pred, zero_division=0),
                        }
                        if "auc_score" in results_df.columns:
                            result["auc_score"] = row["auc_score"]
                        all_results.append(result)

                    except Exception as e:
                        logger.error(
                            f"Error evaluating ensemble {i+1} with method {weight_method}: {e}"
                        )
                        import traceback

                        traceback.print_exc()
                        # Add a result with zero scores to maintain consistency
                        result = {
                            "row_index": row_idx,
                            "ensemble_id": i + 1,
                            "weighting_method": weight_method,
                            "accuracy": 0.0,
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1_score": 0.0,
                        }
                        if "auc_score" in results_df.columns:
                            result["auc_score"] = row["auc_score"]
                        all_results.append(result)

        return pd.DataFrame(all_results)

    def evaluate_on_test_set_from_df(
        self,
        results_df: pd.DataFrame,
        weighting_methods: List[str],
        ensemble_indices: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Evaluates specified ensembles on the TEST set.

        This is a convenience wrapper around `_run_evaluation_from_df` that
        configures the evaluation for the test set.

        Args:
            results_df: The DataFrame containing the GA run results.
            weighting_methods: A list of weighting methods to evaluate.
            ensemble_indices: A list of row indices to evaluate. If None,
                evaluates the single best run by 'auc_score'.
        """
        dataset_name = "TEST SET"  # For logging purposes
        logger.info(
            f"\n--- Evaluating on {dataset_name} for methods: {weighting_methods} ---"
        )
        return self._run_evaluation_from_df(
            results_df,
            weighting_methods,
            use_validation_set=False,
            ensemble_indices=ensemble_indices,
        )

    def validate_on_holdout_set_from_df(
        self,
        results_df: pd.DataFrame,
        weighting_methods: List[str],
        ensemble_indices: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Evaluates specified ensembles on the hold-out VALIDATION set.

        This is a convenience wrapper around `_run_evaluation_from_df` that
        configures the evaluation for the validation set.

        Args:
            results_df: The DataFrame containing the GA run results.
            weighting_methods: A list of weighting methods to evaluate.
            ensemble_indices: A list of row indices to evaluate. If None,
                evaluates the single best run by 'auc_score'.
        """
        dataset_name = "VALIDATION (HOLD-OUT) SET"  # For logging purposes
        logger.info(
            f"\n--- Validating on {dataset_name} for methods: {weighting_methods} ---"
        )
        return self._run_evaluation_from_df(
            results_df,
            weighting_methods,
            use_validation_set=True,
            ensemble_indices=ensemble_indices,
        )


def load_pickled_ensembles(ensemble_path: str) -> Any:
    """Loads a pickled ensemble from a file.

    Args:
        ensemble_path: The path to the pickled ensemble file.

    Returns:
        The unpickled ensemble object.
    """
    with open(ensemble_path, "rb") as f:
        data = pickle.load(f)
    return data
