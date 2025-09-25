import time
import traceback
from typing import Any, Dict
import logging

import keras
import numpy as np
import pandas as pd
from ml_grid.model_classes.keras_classifier_class import kerasClassifier_class
from ml_grid.util.debug_print_statements import debug_print_statements_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.project_score_save import project_score_save_class
from numpy import absolute, mean, std
from scikeras.wrappers import KerasClassifier
from sklearn import metrics
from sklearn.metrics import *
from sklearn.metrics import (
    classification_report,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    RepeatedKFold,
    cross_validate,
)


from ml_grid.util.global_params import global_parameters
logger = logging.getLogger("ensemble_ga")


import warnings

# from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import tensorflow as tf


class grid_search_crossvalidate:
    """
    Performs a grid search or randomized search with cross-validation for a given model.

    This class orchestrates the hyperparameter tuning process for a single
    algorithm. It can perform either an exhaustive `GridSearchCV` or a
    `RandomizedSearchCV` based on the global project settings. After finding
    the best hyperparameters, it performs a final cross-validation on the
    best estimator to get robust performance metrics and logs the results.
    """

    global_params: global_parameters
    """An instance of the global_parameters class."""

    verbose: int
    """The verbosity level, inherited from global_params."""

    sub_sample_param_space_pct: float
    """The percentage of the parameter space to sample in a random grid search."""

    sub_sample_parameter_val: int
    """The number of parameter combinations to sample for random search."""

    metric_list: Dict
    """A dictionary of scoring metrics for cross-validation."""

    error_raise: bool
    """Flag to determine if errors should be raised, from global_params."""

    ml_grid_object_iter: Any
    """The main experiment object, containing data splits and configurations."""

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

    cv: RepeatedKFold
    """The cross-validation strategy object."""

    def __init__(
        self,
        algorithm_implementation: Any,
        parameter_space: Dict,
        method_name: str,
        ml_grid_object: Any,
        sub_sample_parameter_val: int = 100,
    ):
        """Initializes and runs the grid search cross-validation process.

        Args:
            algorithm_implementation: The scikit-learn compatible model instance.
            parameter_space: A dictionary defining the hyperparameter search space.
            method_name: A string name for the method being evaluated.
            ml_grid_object: The main experiment object, containing data and configs.
            sub_sample_parameter_val: The number of parameter combinations to sample
                in a randomized search. Defaults to 100.
        """
        warnings.filterwarnings("ignore")

        warnings.filterwarnings("ignore", category=FutureWarning)

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        warnings.filterwarnings("ignore", category=UserWarning)

        self.global_params = global_parameters()

        self.verbose = self.global_params.verbose

        self.sub_sample_param_space_pct = self.global_params.sub_sample_param_space_pct

        random_grid_search = self.global_params.random_grid_search

        self.sub_sample_parameter_val = sub_sample_parameter_val

        grid_n_jobs = self.global_params.grid_n_jobs

        if "keras" in method_name.lower():
            grid_n_jobs = 1
            gpu_devices = tf.config.experimental.list_physical_devices("GPU")
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)

        if "XGBClassifier" in method_name.lower():
            grid_n_jobs = 1

        self.metric_list = self.global_params.metric_list

        self.error_raise = self.global_params.error_raise

        if self.verbose >= 3:
            logger.info("crossvalidating %s", method_name)

        self.global_parameters = global_parameters()

        self.ml_grid_object_iter = ml_grid_object

        self.X_train = self.ml_grid_object_iter.X_train

        self.y_train = self.ml_grid_object_iter.y_train

        self.X_test = self.ml_grid_object_iter.X_test

        self.y_test = self.ml_grid_object_iter.y_test

        self.X_test_orig = self.ml_grid_object_iter.X_test_orig

        self.y_test_orig = self.ml_grid_object_iter.y_test_orig

        self.cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

        start = time.time()

        current_algorithm = algorithm_implementation

        parameters = parameter_space
        n_iter_v = np.nan
        #     if(sub_sample_param_space):
        #         sub_sample_param_space_n = int(sub_sample_param_space_pct *  len(ParameterGrid(parameter_space)))
        #         parameter_space random.sample(ParameterGrid(parameter_space), sub_sample_param_space_n)

        # Grid search over hyperparameter space, randomised.
        if random_grid_search:
            # n_iter_v = int(self.sub_sample_param_space_pct *  len(ParameterGrid(parameter_space))) + 2
            n_iter_v = int(len(ParameterGrid(parameter_space))) + 2

            if self.sub_sample_parameter_val < n_iter_v:
                n_iter_v = self.sub_sample_parameter_val

            grid = RandomizedSearchCV(
                current_algorithm,
                parameters,
                verbose=1,
                cv=[(slice(None), slice(None))],
                n_jobs=grid_n_jobs,
                n_iter=n_iter_v,
                error_score=np.nan,
            )
        else:
            grid = GridSearchCV(
                current_algorithm,
                parameters,
                verbose=1,
                cv=[(slice(None), slice(None))],
                n_jobs=grid_n_jobs,
                error_score=np.nan,
            )  # Negate CV in param search for speed

        pg = ParameterGrid(parameter_space)
        pg = len(pg)

        if (random_grid_search and n_iter_v > 100000) or (
            random_grid_search == False and pg > 100000
        ):
            logger.error("grid too large %s %s", str(pg), str(n_iter_v))
            raise Exception("grid too large", str(pg))

        if self.global_parameters.verbose >= 1:
            if random_grid_search:
                logger.info("Randomized parameter grid size for %s \n : Full: %s, (mean * %s): %s, current: %s ", current_algorithm, pg, self.sub_sample_param_space_pct, self.sub_sample_parameter_val, n_iter_v)

            else:
                logger.info("parameter grid size: Full: %s", pg)
        grid.fit(self.X_train, self.y_train)

        # Get cross validated scores for best hyperparameter model on x_train_/y_train
        if type(grid.estimator) is not keras.wrappers.scikit_learn.KerasClassifier:

            current_algorithm = grid.best_estimator_
            current_algorithm.fit(self.X_train, self.y_train)

        else:
            current_algorithm = KerasClassifier(
                build_fn=kerasClassifier_class.create_model(),  # dual function definition...in model class.
                verbose=0,
                layers=grid.best_params_["layers"],
                width=grid.best_params_["width"],
                learning_rate=grid.best_params_["learning_rate"],
            )

        scores = cross_validate(
            current_algorithm,
            self.X_train,
            self.y_train,
            scoring=self.metric_list,
            cv=self.cv,
            n_jobs=grid_n_jobs,  # Full CV on final best model #exp -1 was 1
            pre_dispatch=80,  # exp,
            error_score=np.nan,
        )
        current_algorithm_scores = scores
        #     scores_tuple_list.append((method_name, current_algorithm_scores, grid))

        if self.global_parameters.verbose >= 4:

            debug_print_statements_class.debug_print_scores(scores)

        plot_auc = False
        if plot_auc:
            # This was passing a classifier trained on the test dataset....
            logger.info(" ")

            # plot_auc_results(current_algorithm, self.X_test_orig[self.X_train.columns], self.y_test_orig, self.cv)
            # plot_auc_results(grid.best_estimator_, X_test_orig, self.y_test_orig, cv)

        #         this should be x_test...?
        best_pred_orig = current_algorithm.predict(
            self.X_test[self.X_test.columns]
        )  # exp

        project_score_save_class.update_score_log(
            self=self,
            ml_grid_object=self.ml_grid_object_iter,
            scores=scores,
            best_pred_orig=best_pred_orig,
            current_algorithm=current_algorithm,
            method_name=method_name,
            pg=pg,
            start=start,
            n_iter_v=n_iter_v,
        )


#         when to use validation set... and how to store which cases are in this valid set? can withold valid set even earlier...? should?
