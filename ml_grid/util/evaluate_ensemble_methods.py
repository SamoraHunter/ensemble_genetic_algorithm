# A module for evaluating ensembles by calling the prediction resolver function.

import pandas as pd
from ml_grid.pipeline.evaluate_methods_y_pred_resolver import get_y_pred_resolver_eval
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

try:
    from ml_grid.pipeline.data_train_test_split import get_data_split
    from ml_grid.pipeline.evaluate_methods_y_pred_resolver import (
        get_y_pred_resolver_eval,
    )
except ImportError as e:
    print(f"Error: Could not import required functions from 'ml_grid': {e}")

    # Define dummy functions to allow the script to be parsed.
    def get_data_split(*args, **kwargs):
        raise ImportError("ml_grid.pipeline.data_train_test_split not found.")

    def get_y_pred_resolver_eval(*args, **kwargs):
        raise ImportError("ml_grid.pipeline.evaluate_methods_ga not found.")


# --- Helper Class to Mimic MLGridObject ---
class MLGridObject:
    """A container class to hold data and parameters for the resolver function."""

    def __init__(self):
        self.verbose = 1
        self.local_param_dict = {}
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.X_test_orig, self.y_test_orig = None, None  # This is the validation set


class EnsembleEvaluator:
    """
    Evaluates ensembles by calling an external prediction resolver function.
    Now uses a DataFrame with results and the 'best_ensemble' column, defaults to top by auc_score.
    """

    def __init__(
        self,
        input_csv_path: str,
        outcome_variable: str,
        initial_param_dict: dict,
        debug: bool = False,
    ):
        self.debug = debug
        if self.debug:
            print("--- Initializing EnsembleEvaluator ---")
        self.ml_grid_object = MLGridObject()
        self.ml_grid_object.local_param_dict = initial_param_dict
        self.ml_grid_object.verbose = 0
        self.original_feature_names = None
        self._load_and_split_data(input_csv_path, outcome_variable)

    def _load_and_split_data(self, input_csv_path, outcome_variable):
        if self.debug:
            print(f"Loading data from: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        y = df[outcome_variable]
        X = df.drop(outcome_variable, axis=1)
        self.original_feature_names = list(X.columns)
        if self.debug:
            print("Splitting data and assigning to ml_grid_object...")
        X_train, X_test, y_train, y_test, X_val, y_val = get_data_split(
            X, y, self.ml_grid_object.local_param_dict
        )
        self.ml_grid_object.X_train, self.ml_grid_object.y_train = X_train, y_train
        self.ml_grid_object.X_test, self.ml_grid_object.y_test = X_test, y_test
        self.ml_grid_object.X_test_orig, self.ml_grid_object.y_test_orig = X_val, y_val
        if self.debug:
            print("Data splits assigned successfully.")

    def _parse_ensemble(self, ensemble_record, debug=False):
        """Parse a best_ensemble record into a list of processed ensembles."""
        import pandas as pd
        from numpy import array

        # If it's a pandas Series, extract the first value
        if isinstance(ensemble_record, pd.Series):
            if debug:
                print(
                    f"[DEBUG] ensemble_record is a Series, extracting first value: {ensemble_record}"
                )
            ensemble_record = ensemble_record.iloc[0]
        # If it's a string, eval it; if not, assume it's already a list/tuple
        if isinstance(ensemble_record, str):
            try:
                ensembles = eval(ensemble_record, {"array": array})
            except Exception as e:
                if debug:
                    print(f"Could not eval ensemble_record: {e}")
                return []
        else:
            ensembles = ensemble_record
        if debug:
            print(f"[DEBUG] ensembles after eval: {ensembles}")
        processed_ensembles = []
        # If ensembles is a Series, extract first value
        if hasattr(ensembles, "iloc") and not isinstance(ensembles, (list, tuple)):
            if debug:
                print(
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
                    print(f"[DEBUG] Skipping non-list/tuple ensemble: {ensemble}")
                continue
            processed_ensemble = []
            for model_tuple in ensemble:
                if not isinstance(model_tuple, (list, tuple)) or len(model_tuple) < 2:
                    if debug:
                        print(
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
                        print(
                            f"Could not eval model string: '{model_string}'. Error: {e}"
                        )
                    continue
            processed_ensembles.append(processed_ensemble)
        return processed_ensembles

    def _run_evaluation_from_df(
        self,
        results_df: pd.DataFrame,
        weighting_methods: list,
        use_validation_set: bool = False,
        ensemble_indices: list = None,
    ) -> pd.DataFrame:
        """
        Core evaluation loop for a given set of weighting methods using a results DataFrame.
        By default, processes the top ensemble by auc_score.
        ensemble_indices: list of row indices (or single int) to evaluate, or 'all' for all rows.
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
                print(
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
                            print(
                                f"[DEBUG] y_true length: {len(y_true)}, y_pred length: {len(y_pred)}"
                            )
                            print(
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
                        print(
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
        weighting_methods: list,
        ensemble_indices: list = None,
    ) -> pd.DataFrame:
        """
        Evaluates ensembles on the TEST set for each weighting method using a results DataFrame.
        ensemble_indices: which row(s) to evaluate (default: top by auc_score).
        """
        dataset_name = "TEST SET"
        print(
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
        weighting_methods: list,
        ensemble_indices: list = None,
    ) -> pd.DataFrame:
        """
        Evaluates ensembles on the VALIDATION set for each weighting method using a results DataFrame.
        ensemble_indices: which row(s) to evaluate (default: top by auc_score).
        """
        dataset_name = "VALIDATION (HOLD-OUT) SET"
        print(
            f"\n--- Validating on {dataset_name} for methods: {weighting_methods} ---"
        )
        return self._run_evaluation_from_df(
            results_df,
            weighting_methods,
            use_validation_set=True,
            ensemble_indices=ensemble_indices,
        )


def load_pickled_ensembles(ensemble_path):
    """
    Loads a pickled ensemble from a file.

    Parameters
    ----------
    ensemble_path : str
        The path to the pickled ensemble file.

    Returns
    -------
    data : object
        The pickled ensemble object.
    """
    with open(ensemble_path, "rb") as f:
        data = pickle.load(f)
    return data
