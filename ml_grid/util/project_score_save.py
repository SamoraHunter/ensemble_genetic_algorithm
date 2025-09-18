import os
import time
import traceback
from typing import Any, Dict, List
import warnings

import numpy as np
import pandas as pd
from ml_grid.util.global_params import global_parameters
from sklearn import metrics

# from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import *


class project_score_save_class:
    """Manages the logging of experiment results to a master CSV file.

    This class is responsible for initializing a CSV log file with a predefined
    set of headers and provides a method to append new results from each
    experimental run. It handles the calculation of various performance metrics
    and the extraction of configuration parameters.
    """

    global_params: global_parameters
    """An instance of the global_parameters class."""

    metric_list: Dict
    """A dictionary of scoring metrics, inherited from global_params."""

    error_raise: bool
    """Flag to determine if errors should be raised, from global_params."""

    def __init__(self, base_project_dir):

        warnings.filterwarnings("ignore")

        warnings.filterwarnings("ignore", category=FutureWarning)

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        warnings.filterwarnings("ignore", category=UserWarning)

        self.global_params = global_parameters()

        self.metric_list = self.global_params.metric_list

        self.error_raise = self.global_params.error_raise

        # init final grid scores
        column_list = [
            "nb_size",
            "f_list",
            "auc",
            "mcc",
            "f1",
            "precision",
            "recall",
            "accuracy",
            "nb_val",
            "pop_val",
            "g_val",
            "g",
            "weighted",
            "use_stored_base_learners",
            "store_base_learners",
            "resample",
            "scale",
            "n_features",
            "param_space_size",
            "n_unique_out",
            "outcome_var_n",
            "div_p",
            "percent_missing",
            "corr",
            "age",
            "sex",
            "bmi",
            "ethnicity",
            "bloods",
            "diagnostic_order",
            "drug_order",
            "annotation_n",
            "meta_sp_annotation_n",
            "meta_sp_annotation_mrc_n",
            "annotation_mrc_n",
            "core_02",
            "bed",
            "vte_status",
            "hosp_site",
            "core_resus",
            "news",
            "date_time_stamp",
            "X_train_size",
            "X_test_orig_size",
            "X_test_size",
            "run_time",
            "cxpb",
            "mutpb",
            "indpb",
            "t_size",
            "valid",
            "generation_progress_list",
            "best_ensemble",
        ]

        # column_list = column_list + ["BL_" + str(x) for x in range(0, 64)]

        df = pd.DataFrame(data=None, columns=column_list)

        file_path = base_project_dir + "final_grid_score_log.csv"

        if not os.path.exists(file_path):
            df.to_csv(
                file_path,
                mode="w",
                header=True,
                index=False,
            )

    def update_score_log(
        self,
        ml_grid_object: Any,
        scores: Dict,
        best_pred_orig: np.ndarray,
        current_algorithm: Any,
        method_name: str,
        pg: int,
        start: float,
        n_iter_v: int,
        valid: bool = False,
        generation_progress_list: List = [],
        best_ensemble: str = "",
    ):
        """Calculates metrics and appends a new result row to the log file.

        This method takes the results of a single experimental run, calculates
        a comprehensive set of performance metrics (AUC, MCC, F1, etc.),
        extracts all relevant configuration parameters from the `ml_grid_object`,
        and writes this information as a new row in the project's master
        `final_grid_score_log.csv` file.

        Args:
            ml_grid_object: The main experiment object, containing data splits
                and configuration parameters.
            scores: A dictionary of cross-validation scores.
            best_pred_orig: The predictions from the best-performing model/ensemble.
            current_algorithm: The best model or ensemble object.
            method_name: A string identifier for the method used.
            pg: The size of the parameter grid searched.
            start: The `time.time()` timestamp from the beginning of the run.
            n_iter_v: The number of iterations performed in the search.
            valid: If True, evaluation is performed on the validation set
                (`y_test_orig`). Otherwise, it's on the test set (`y_test`).
            generation_progress_list: A list of fitness scores per generation (for GA).
            best_ensemble: A string representation of the final best ensemble.
        """

        if ml_grid_object.verbose >= 1:
            print("update_score_log")
            # Debugging messages and variable prints
            print("Valid:", valid)
            print("ML grid object:", ml_grid_object)
            print("Scores:", scores)
            print("Best prediction original:", best_pred_orig)
            print("Current algorithm:", current_algorithm)
            print("Method name:", method_name)
            print("PG:", pg)
            print("Start:", start)
            print("Number of iterations:", n_iter_v)

        self.global_parameters = global_parameters()

        self.ml_grid_object_iter = ml_grid_object

        self.X_train = self.ml_grid_object_iter.X_train

        self.y_train = self.ml_grid_object_iter.y_train

        self.X_test = self.ml_grid_object_iter.X_test

        self.y_test = self.ml_grid_object_iter.y_test

        self.X_test_orig = self.ml_grid_object_iter.X_test_orig

        self.y_test_orig = self.ml_grid_object_iter.y_test_orig

        self.param_space_index = self.ml_grid_object_iter.param_space_index
        # n_iter_v = np.nan ##????????????

        if ml_grid_object.verbose >= 1:
            # Print shapes of data
            print("X_train shape:", self.X_train.shape)
            print("y_train shape:", self.y_train.shape)
            print("X_test shape:", self.X_test.shape)
            print("y_test shape:", self.y_test.shape)
            print("X_test_orig shape:", self.X_test_orig.shape)
            print("y_test_orig shape:", self.y_test_orig.shape)
            print("Global parameters:", self.global_parameters)
            print("best_pred_orig len", len(best_pred_orig))
        try:
            print("Writing grid permutation to log")
            # write line to best grid scores---------------------
            column_list = [
                "nb_size",
                "f_list",
                "auc",
                "mcc",
                "f1",
                "precision",
                "recall",
                "accuracy",
                "nb_val",
                "pop_val",
                "g_val",
                "g",
                "weighted",
                "use_stored_base_learners",
                "store_base_learners",
                "resample",
                "scale",
                "n_features",
                "param_space_size",
                "n_unique_out",
                "outcome_var_n",
                "div_p",
                "percent_missing",
                "corr",
                "age",
                "sex",
                "bmi",
                "ethnicity",
                "bloods",
                "diagnostic_order",
                "drug_order",
                "annotation_n",
                "meta_sp_annotation_n",
                "meta_sp_annotation_mrc_n",
                "annotation_mrc_n",
                "core_02",
                "bed",
                "vte_status",
                "hosp_site",
                "core_resus",
                "news",
                "date_time_stamp",
                "X_train_size",
                "X_test_orig_size",
                "X_test_size",
                "run_time",
                "cxpb",
                "mutpb",
                "indpb",
                "t_size",
                "valid",
                "generation_progress_list",
                "best_ensemble",
            ]

            # column_list = column_list + ["BL_" + str(x) for x in range(0, 64)]

            line = pd.DataFrame(data=None, columns=column_list)

            if valid:
                y_true = self.y_test_orig
                debug_message = "Using self.y_test_orig for evaluation."
            else:
                y_true = self.y_test
                debug_message = "Using self.y_test for evaluation."
            if ml_grid_object.verbose > 1:
                print(debug_message)
            auc = metrics.roc_auc_score(y_true, best_pred_orig)
            mcc = matthews_corrcoef(y_true, best_pred_orig)
            f1 = f1_score(y_true, best_pred_orig, average="binary")
            precision = precision_score(y_true, best_pred_orig, average="binary")
            recall = recall_score(y_true, best_pred_orig, average="binary")
            accuracy = accuracy_score(y_true, best_pred_orig)

            # get info from current settings iter...local_param_dict ml_grid_object
            for key in ml_grid_object.local_param_dict:

                if key != "data":
                    if key in column_list:
                        line[key] = [ml_grid_object.local_param_dict.get(key)]
                else:
                    for key_1 in ml_grid_object.local_param_dict.get("data"):

                        if key_1 in column_list:
                            line[key_1] = [
                                ml_grid_object.local_param_dict.get("data").get(key_1)
                            ]

            current_f = list(self.X_test.columns)
            current_f_vector = []
            f_list = []
            for elem in ml_grid_object.orignal_feature_names:
                if elem in current_f:
                    current_f_vector.append(1)
                else:
                    current_f_vector.append(0)

            f_list.append(current_f_vector)

            line["algorithm_implementation"] = [current_algorithm]
            line["parameter_sample"] = [current_algorithm]
            line["method_name"] = [method_name]
            line["nb_size"] = [sum(np.array(current_f_vector))]
            line["n_features"] = [len(current_f_vector)]
            line["f_list"] = [f_list]

            line["auc"] = [auc]
            line["mcc"] = [mcc]
            line["f1"] = [f1]
            line["precision"] = [precision]
            line["recall"] = [recall]
            line["accuracy"] = [accuracy]

            line["X_train_size"] = [len(self.X_train)]
            line["X_test_orig_size"] = [len(self.X_test_orig)]
            line["X_test_size"] = [len(self.X_test)]

            end = time.time()

            line["run_time"] = int((end - start) / 60)
            line["t_fits"] = pg
            line["n_fits"] = n_iter_v
            line["i"] = self.param_space_index  # 0 # should be index of the iterator

            line["nb_val"] = [ml_grid_object.nb_val]
            line["pop_val"] = [ml_grid_object.pop_val]
            line["g_val"] = [ml_grid_object.g_val]
            line["g"] = [ml_grid_object.g]
            line["generation_progress_list"] = [generation_progress_list]
            line["best_ensemble"] = [str(best_ensemble)]

            if ml_grid_object.verbose >= 1:
                print("current_algorithm")
                print(current_algorithm)

            # for iii in range(0, len(current_algorithm[0])):
            #     f_list = []
            #     current_f = current_algorithm[0][iii][2]

            #     current_f_vector = []
            #     for elem in ml_grid_object.orignal_feature_names:
            #         if elem in current_f:
            #             current_f_vector.append(1)
            #         else:
            #             current_f_vector.append(0)

            #     f_list.append(current_f_vector)

            #     line["BL_" + str(iii)] = [f_list]

            print(line)

            line[column_list].to_csv(
                ml_grid_object.base_project_dir + "final_grid_score_log.csv",
                mode="a",
                header=False,
                index=True,
            )

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print("Failed to upgrade grid entry")
            raise
            if self.error_raise:
                input()
