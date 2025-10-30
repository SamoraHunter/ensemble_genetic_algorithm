import logging
import traceback
from typing import Any, Dict, List

import numpy as np
from sklearn.model_selection import ParameterGrid

from ml_grid.model_classes.adaboost_classifier_class import adaboost_class
from ml_grid.model_classes.gaussiannb_class import GaussianNB_class
from ml_grid.model_classes.gradientboosting_classifier_class import (
    GradientBoostingClassifier_class,
)
from ml_grid.model_classes.keras_classifier_class import kerasClassifier_class
from ml_grid.model_classes.knn_classifier_class import knn_classifiers_class
from ml_grid.model_classes.logistic_regression_class import LogisticRegression_class
from ml_grid.model_classes.mlp_classifier_class import mlp_classifier_class
from ml_grid.model_classes.quadratic_discriminant_class import (
    quadratic_discriminant_analysis_class,
)
from ml_grid.model_classes.randomforest_classifier_class import (
    RandomForestClassifier_class,
)
from ml_grid.model_classes.svc_class import SVC_class
from ml_grid.model_classes.xgb_classifier_class import XGB_class_class

# from ml_grid.model_classes import LogisticRegression_class
from ml_grid.pipeline import grid_search_cross_validate
from ml_grid.util.global_params import global_parameters

logger = logging.getLogger("ensemble_ga")


class run:
    """Orchestrates a grid search cross-validation for a list of predefined models.

    This class initializes a suite of model classes, each with its own
    hyperparameter space. It then prepares arguments for each model to be
    passed to a grid search function. The `execute` method iterates through
    these models and runs the grid search, handling errors and logging
    the outcomes. This class is designed for a more traditional grid search
    approach, as opposed to the genetic algorithm pipeline.
    """

    global_params: global_parameters
    """An instance of the `global_parameters` class."""

    verbose: int
    """The verbosity level, inherited from global_params."""

    error_raise: bool
    """A flag to determine if errors should be raised, from `global_params`."""

    ml_grid_object: Any
    """The main experiment object, containing data splits and configurations."""

    sub_sample_param_space_pct: float
    """The percentage of the parameter space to sample for a random grid search."""

    parameter_space_size: str
    """The size of the parameter space to use (e.g., 'medium', 'xsmall')."""

    model_class_list: List[Any]
    """A list of the instantiated model classes to be evaluated."""

    pg_list: List[int]
    """A list containing the size of the parameter grid for each model."""

    mean_parameter_space_val: float
    """The mean size of the parameter grids across all evaluated models."""

    sub_sample_parameter_val: int
    """The number of parameter combinations to sample for random search."""

    arg_list: List[tuple]
    """A list of argument tuples for the `grid_search_crossvalidate` function."""

    multiprocess: bool
    """A flag to enable or disable multiprocessing (currently disabled)."""

    local_param_dict: Dict
    """A dictionary of local parameters for the current run."""

    model_error_list: List[List]
    """A list to store any errors encountered during model evaluation."""

    def __init__(self, ml_grid_object: Any, local_param_dict: Dict):
        """Initializes the grid search runner.

        This constructor sets up the environment, loads global parameters, and a
        initializes a list of all model classes that will be subjected to
        grid search. It calculates the size of each model's parameter space
        and prepares the arguments for the `grid_search_crossvalidate` function.

        Args:
            ml_grid_object: The main experiment object, containing data splits
                and configurations.
            local_param_dict: A dictionary of local parameters for the current run.
        """
        self.global_params = global_parameters()

        self.verbose = self.global_params.verbose

        self.error_raise = self.global_params.error_raise

        self.ml_grid_object = ml_grid_object

        self.sub_sample_param_space_pct = self.global_params.sub_sample_param_space_pct

        self.parameter_space_size = local_param_dict.get("param_space_size")

        self.model_class_list = [
            LogisticRegression_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            knn_classifiers_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            quadratic_discriminant_analysis_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            SVC_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            XGB_class_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            mlp_classifier_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            RandomForestClassifier_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            GradientBoostingClassifier_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            kerasClassifier_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            GaussianNB_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            adaboost_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            # knn__gpu_wrapper_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train, parameter_space_size=self.parameter_space_size),
        ]

        if self.verbose >= 2:
            logger.info("%s models loaded", len(self.model_class_list))

        self.pg_list = []

        for elem in self.model_class_list:

            pg = ParameterGrid(elem.parameter_space)

            self.pg_list.append(len(ParameterGrid(elem.parameter_space)))

            if self.verbose >= 1:
                logger.info("%s:%s", elem.method_name, len(pg))

            for param in elem.parameter_space:
                try:
                    if type(param) is not list:
                        if (
                            isinstance(elem.parameter_space.get(param), list) is False
                            and isinstance(elem.parameter_space.get(param), np.ndarray)
                            is False
                        ):
                            logger.warning("What is this?")
                            logger.warning("%s, %s %s", elem.method_name, param, type(elem.parameter_space.get(param)))

                except Exception:
                    # logger.debug(e)
                    pass

        # sample from mean of all param space n
        self.mean_parameter_space_val = np.mean(self.pg_list)

        self.sub_sample_parameter_val = int(
            self.sub_sample_param_space_pct * self.mean_parameter_space_val
        )

        # n_iter_v = int(sub_sample_param_space_pct *  len(ParameterGrid(parameter_space)))

        self.arg_list = []
        for model_class in self.model_class_list:

            class_name = model_class

            self.arg_list.append(
                (
                    class_name.algorithm_implementation,
                    class_name.parameter_space,
                    class_name.method_name,
                    self.ml_grid_object,
                    self.sub_sample_parameter_val,
                )
            )

        self.multiprocess = False

        self.local_param_dict = local_param_dict

        if self.verbose >= 2:
            logger.info("Passed main init, len(arg_list): %s", len(self.arg_list))

    def execute(self) -> List[List]:
        """Executes the grid search for all configured models.

        This method iterates through the `arg_list` and calls the
        `grid_search_crossvalidate` function for each model. It logs any
        exceptions that occur during the process.

        Note:
            The multiprocessing functionality is currently disabled.

        Returns:
            A list of errors encountered during the execution. Each item in the
            list contains the model implementation, the exception, and a traceback.
        """
        # needs implementing*

        self.model_error_list = []

        if self.multiprocess:

            def multi_run_wrapper(args):
                logger.warning("not implemented ")
                # return grid_search_cross_validate(*args)

            if __name__ == "__main__":
                from multiprocessing import Pool

                with Pool(8) as pool:
                    pool.map(multi_run_wrapper, self.arg_list)

                pool.close()  # exp

        elif not self.multiprocess:
            for k in range(0, len(self.arg_list)):
                try:
                    logger.info("grid searching...")
                    grid_search_cross_validate.grid_search_crossvalidate(
                        *self.arg_list[k]
                        # algorithm_implementation = LogisticRegression_class(parameter_space_size=self.parameter_space_size).algorithm_implementation, parameter_space = self.arg_list[k][1], method_name=self.arg_list[k][2], X = self.arg_list[k][3], y=self.arg_list[k][4]
                    )
                except Exception as e:

                    logger.error(e)
                    logger.error("error on %s", self.arg_list[k][2])
                    self.model_error_list.append(
                        [self.arg_list[k][0], e, traceback.print_exc()]
                    )

                    if self.error_raise:
                        input(
                            "error thrown in grid_search_crossvalidate on model class list"
                        )

        logger.info("Model error list: nb. errors returned from func: %s", self.model_error_list)
        logger.info(self.model_error_list)

        return self.model_error_list
