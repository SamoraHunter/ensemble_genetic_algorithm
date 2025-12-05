import logging
import os
import random
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from ml_grid.model_classes_ga.logistic_regression_model import (
    logisticRegressionModelGenerator,
)
from ml_grid.pipeline import read_in
from ml_grid.pipeline.column_names import get_pertubation_columns
from ml_grid.pipeline.data_clean_up import clean_up_class
from ml_grid.pipeline.data_constant_columns import (
    remove_constant_columns,
    remove_constant_columns_with_debug,
)
from ml_grid.pipeline.data_correlation_matrix import handle_correlation_matrix
from ml_grid.pipeline.data_feature_importance_methods import feature_importance_methods
from ml_grid.pipeline.data_outcome_list import handle_outcome_list
from ml_grid.pipeline.data_percent_missing import handle_percent_missing
from ml_grid.pipeline.data_plot_split import plot_pie_chart_with_counts
from ml_grid.pipeline.data_train_test_split import get_data_split
from ml_grid.pipeline.logs_project_folder import log_folder
from ml_grid.util.global_params import global_parameters
from ml_grid.util.project_score_save import project_score_save_class

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("ensemble_ga")


class NoFeaturesError(Exception):
    """Custom exception raised when no features are left after processing."""

    pass


class pipe:
    """The main data processing pipeline for an ml_grid experiment.

    This class orchestrates the entire data preparation process, from reading the
    raw data to splitting it into final training, testing, and validation sets.
    It takes a configuration dictionary and applies a series of cleaning and
    feature selection steps.

    The pipeline steps include:
    1.  Reading the input data, with options for sampling rows and columns.
    2.  Categorizing features and selecting a subset for the experiment based on
        the configuration.
    3.  Building a `drop_list` of columns to be removed based on:
        - Predefined terms (`drop_term_list`).
        - High correlation between features.
        - High percentage of missing values.
        - Constant values (zero variance).
        - Ensuring only the target outcome variable is kept.
    4.  Applying the `drop_list` to create the final feature set.
    5.  Cleaning the resulting DataFrame (e.g., handling duplicated columns,
        sanitizing column names).
    6.  Optionally scaling the data.
    7.  Splitting the cleaned data into train, test, and validation sets.
    8.  Optionally performing a final feature selection step on the split data.

    The resulting `pipe` object holds all the data splits and configurations,
    ready to be passed to a model training/evaluation class like `main.run` or
    `main_ga.run`.
    """

    testing: bool
    """Flag to indicate if the pipeline is running in a testing/debug mode."""

    multiprocessing_ensemble: bool
    """Flag to enable multiprocessing for ensemble generation."""

    base_project_dir: str
    """The root directory for the project, used for saving logs and artifacts."""

    additional_naming: Optional[str]
    """An optional string to append to log folder names for easier identification."""

    local_param_dict: Dict
    """A dictionary of parameters for this specific pipeline run."""

    global_params: global_parameters
    """An instance of the global_parameters class holding project-wide settings."""

    verbose: int
    """The verbosity level for console output."""

    param_space_index: int
    """The index of the current parameter configuration being run."""

    project_score_save_object: project_score_save_class
    """An object for saving final scores to the master log file."""

    config_dict: Dict
    """A dictionary of configuration options for the GA, like `use_stored_base_learners`."""

    logging_paths_obj: log_folder
    """An object that manages the creation and paths of logging directories."""

    df: pd.DataFrame
    """The main DataFrame holding the data throughout the processing steps."""

    all_df_columns: List[str]
    """A list of all column names from the initial raw DataFrame."""

    original_feature_names: List[str]
    """A copy of the initial column names, preserved for reference."""

    pertubation_columns: List[str]
    """A list of columns selected for the experiment based on `local_param_dict`."""

    drop_list: List[str]
    """A list of columns to be removed during the cleaning process."""

    outcome_variable: str
    """The name of the target variable column."""

    final_column_list: List[str]
    """The final list of feature columns after all cleaning and selection steps."""

    X: pd.DataFrame
    """The final features DataFrame before splitting."""

    y: pd.Series
    """The final target variable Series before splitting."""

    X_train: pd.DataFrame
    """The training features DataFrame."""

    X_test: pd.DataFrame
    """The testing features DataFrame (for GA evaluation)."""

    y_train: pd.Series
    """The training target Series."""

    y_test: pd.Series
    """The testing target Series (for GA evaluation)."""

    X_test_orig: pd.DataFrame
    """The hold-out validation features DataFrame."""

    y_test_orig: pd.Series
    """The hold-out validation target Series."""

    model_class_list: List
    """A list of model generator functions to be used (for GA)."""

    feature_transformation_log: pd.DataFrame
    """A DataFrame that logs the changes to the feature set at each pipeline step."""

    def __init__(
        self,
        global_params: global_parameters,
        file_name: str,
        drop_term_list: List[str],
        local_param_dict: Dict,
        base_project_dir: str,
        param_space_index: int,
        additional_naming: Optional[str] = None,
        test_sample_n: int = 0,
        column_sample_n: int = 0,
        config_dict: Optional[Dict] = None,
        testing: bool = False,
        multiprocessing_ensemble: bool = False,
    ):
        """Initializes and executes the data processing pipeline.

        Args:
            global_params: An initialized global_parameters object.
            file_name: The path to the input data CSV file.
            drop_term_list: A list of substrings to identify columns for removal.
            local_param_dict: A dictionary of parameters for this specific run.
            base_project_dir: The root directory for the project.
            param_space_index: The index of the current parameter configuration.
            additional_naming: An optional string to append to log folder names.
            test_sample_n: The number of rows to sample for testing. 0 means all.
            column_sample_n: The number of columns to sample. 0 means all.
            config_dict: A dictionary of configuration options for the GA.
            testing: If True, runs in a testing mode (e.g., smaller grids).
            multiprocessing_ensemble: If True, enables multiprocessing for
                ensemble generation.
        """

        self.testing = testing
        self.multiprocessing_ensemble = multiprocessing_ensemble
        self.base_project_dir = os.path.join(base_project_dir, "")
        self.additional_naming = additional_naming
        self.local_param_dict = local_param_dict
        self.global_params = global_params
        self.verbose = self.global_params.verbose
        self.param_space_index = param_space_index

        self.project_score_save_object = project_score_save_class(
            base_project_dir=self.base_project_dir
        )

        self.config_dict = config_dict
        if self.config_dict is None:
            self.config_dict = {
                "use_stored_base_learners": False,
            }
            if self.verbose >= 1:
                logger.info("Using default config_dict... %s", self.config_dict)
        # Ensure modelFuncList is always populated from global_params
        self.config_dict["modelFuncList"] = self.global_params.model_list

        if self.verbose >= 1:
            logger.info("Starting... %s", self.local_param_dict)

        self.logging_paths_obj = log_folder(
            local_param_dict=local_param_dict,
            additional_naming=additional_naming,
            base_project_dir=self.base_project_dir,
        )

        # Initialize feature transformation log
        self._feature_log_list = []

        # Execute pipeline with error handling
        pipeline_error = None
        try:
            self._load_data(file_name, test_sample_n, column_sample_n)
            self._initial_feature_selection(drop_term_list)
            self._apply_safety_net()
            self._create_xy()
            self._split_data()
            self._post_split_cleaning()
            self._scale_features()
            self._select_features_by_importance()
            self._finalize_pipeline()
        except Exception as e:
            pipeline_error = e
            raise
        finally:
            self._compile_and_log_feature_transformations(
                error_occurred=pipeline_error is not None
            )
            if pipeline_error:
                logger.error("Data pipeline processing HALTED due to an error.")
            else:
                logger.info("Data pipeline processing complete.")

    def _log_feature_transformation(
        self, step_name: str, before_count: int, after_count: int, description: str
    ):
        """Helper function to log feature transformation steps."""
        if self.verbose >= 1:
            self._feature_log_list.append(
                {
                    "step": step_name,
                    "features_before": before_count,
                    "features_after": after_count,
                    "features_changed": before_count - after_count,
                    "description": description,
                }
            )

    def _assert_index_alignment(
        self, df1: pd.DataFrame, df2: pd.Series, step_name: str
    ):
        """Helper function to assert that DataFrame and Series indices are equal."""
        try:
            assert_index_equal(df1.index, df2.index)
            logger.debug(f"Index alignment PASSED at: {step_name}")
        except AssertionError:
            logger.error(f"Index alignment FAILED at: {step_name}")
            raise

    def _load_data(self, file_name: str, test_sample_n: int, column_sample_n: int):
        """Loads data from the source file."""
        read_in_sample = True

        if read_in_sample and test_sample_n > 0 or column_sample_n > 0:
            self.df = read_in.read_sample(
                file_name, test_sample_n, column_sample_n
            ).raw_input_data
        else:
            self.df = read_in.read(file_name, use_polars=True).raw_input_data

        if test_sample_n > 0 and not read_in_sample:
            logger.info("sampling %s for debug/trial purposes...", test_sample_n)
            self.df = self.df.sample(test_sample_n)

        if column_sample_n > 0 and not read_in_sample:
            # Check if 'age' and 'male' columns are in the original DataFrame
            if (
                "age" in self.df.columns
                and "male" in self.df.columns
                and "outcome_var_1" in self.df.columns
            ):
                original_columns = ["age", "male", "outcome_var_1"]
            else:
                original_columns = []

            logger.info(
                "Sampling %s columns for additional debug/trial purposes...",
                column_sample_n,
            )

            # Sample the columns
            sampled_columns = self.df.sample(n=column_sample_n, axis=1).columns

            # Ensure original columns are retained
            new_columns = list(set(sampled_columns) | set(original_columns))

            # Reassign DataFrame with sampled columns
            self.df = self.df[new_columns].copy()

            logger.info("Result df shape %s", self.df.shape)

        self.all_df_columns = list(self.df.columns)
        self.original_feature_names = self.all_df_columns.copy()
        self._log_feature_transformation(
            "Initial Load",
            len(self.all_df_columns),
            len(self.all_df_columns),
            "Initial data loaded.",
        )

    def _initial_feature_selection(self, drop_term_list: List[str]):
        """Performs initial feature selection based on configuration."""
        # Disable fallback if all data toggles are False, to test the safety net.
        self.pertubation_columns, self.drop_list = get_pertubation_columns(
            all_df_columns=self.all_df_columns,
            local_param_dict=self.local_param_dict,
            drop_term_list=drop_term_list,
        )

        self.outcome_variable = (
            f'outcome_var_{self.local_param_dict.get("outcome_var_n")}'
        )

        logger.info(
            f"Using {len(self.pertubation_columns)}/{len(self.all_df_columns)} columns for {self.outcome_variable} outcome"
        )

        self._log_feature_transformation(
            "Feature Selection (Toggles)",  # This step's features_after count is based on pertubation_columns
            # The drop_list here is the one returned by get_pertubation_columns, which is used to filter pertubation_columns
            # So, features_after should reflect the size of pertubation_columns
            len(self.all_df_columns),
            len(self.pertubation_columns),
            "Selected columns based on feature toggles in config.",
        )

        # Log omitted columns
        difference_list = list(set(self.df.columns) - set(self.pertubation_columns))
        logger.info(
            "Omitting %s columns based on feature toggles...", len(difference_list)
        )
        logger.debug("Sample of omitted columns: %s...", difference_list[0:5])

        # Initialize self.drop_list with unique columns identified so far
        # self.drop_list from get_pertubation_columns contains columns matching drop_term_list
        self.drop_list = list(set(self.drop_list))

        # Apply correlation matrix filtering and add to drop_list
        # Apply correlation matrix filtering
        features_before = len(self.pertubation_columns)
        # handle_correlation_matrix returns a list of columns to drop due to correlation.
        # It does not modify the passed drop_list.
        logger.debug(
            "Dropping columns with correlation > %s",
            self.local_param_dict.get("corr", 0.95),
        )
        correlated_drops = handle_correlation_matrix(
            local_param_dict=self.local_param_dict, drop_list=self.drop_list, df=self.df
        )
        logger.debug("Correlated drops: %s", correlated_drops)

        self.drop_list.extend(correlated_drops)
        self.drop_list = list(
            set(self.drop_list)
        )  # Ensure uniqueness after adding correlated drops
        features_after = len(
            [col for col in self.pertubation_columns if col not in self.drop_list]
        )
        self._log_feature_transformation(
            "Drop Correlated",
            features_before,
            features_after,
            f"Dropped columns with correlation > {self.local_param_dict.get('corr', 0.95)}",
        )

        # Apply percent missing filtering and add to drop_list
        # Apply percent missing filtering
        features_before = features_after
        # handle_percent_missing modifies the passed drop_list in place and returns it.
        # So, we reassign self.drop_list to capture any new unique drops.
        self.drop_list = handle_percent_missing(  # This will extend self.drop_list and return it
            local_param_dict=self.local_param_dict,
            all_df_columns=self.all_df_columns,
            drop_list=self.drop_list,  # Pass the current comprehensive self.drop_list
        )
        self.drop_list = list(set(self.drop_list))  # Ensure uniqueness
        features_after = len(
            [col for col in self.pertubation_columns if col not in self.drop_list]
        )
        self._log_feature_transformation(
            "Drop Missing",
            features_before,
            features_after,
            f"Dropped columns with > {self.local_param_dict.get('percent_missing', 100)}% missing",
        )

        # Remove other outcome variables and add to drop_list
        # Remove other outcome variables
        features_before = features_after
        # handle_outcome_list modifies the passed drop_list in place and returns it.
        self.drop_list = (
            handle_outcome_list(  # This will extend self.drop_list and return it
                drop_list=self.drop_list, outcome_variable=self.outcome_variable
            )
        )
        self.drop_list = list(set(self.drop_list))  # Ensure uniqueness
        features_after = len(
            [col for col in self.pertubation_columns if col not in self.drop_list]
        )
        self._log_feature_transformation(
            "Drop Other Outcomes",
            features_before,
            features_after,
            "Removed other potential outcome variables from feature set.",
        )

        # Remove constant columns and add to drop_list
        # Remove constant columns
        features_before = features_after
        # remove_constant_columns modifies the passed drop_list in place and returns it.
        self.drop_list = (
            remove_constant_columns(  # This will extend self.drop_list and return it
                X=self.df, drop_list=self.drop_list, verbose=self.verbose
            )
        )
        self.drop_list = list(set(self.drop_list))  # Ensure uniqueness
        self._log_feature_transformation(
            "Drop Constants",
            features_before,
            len([col for col in self.pertubation_columns if col not in self.drop_list]),
            "Removed constant columns.",
        )

        self.final_column_list = [
            col for col in self.pertubation_columns if col not in self.drop_list
        ]

    def _apply_safety_net(self):
        """Retains a minimal set of features if all have been pruned."""
        if not self.final_column_list:
            logger.warning("All features pruned! Activating safety retention...")

            # Get a list of all original features that are numeric and not the outcome variable.
            numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            potential_features = [
                c
                for c in self.original_feature_names  # Iterate over all original features
                if c in numeric_cols
                and c != self.outcome_variable
                and c not in self.drop_list
            ]

            safety_columns = []
            if len(potential_features) >= 2:
                safety_columns = random.sample(potential_features, 2)
            elif potential_features:
                safety_columns = potential_features
            else:
                logger.error("No features available to select for safety net.")

            self.final_column_list = safety_columns
            logger.info("Retaining minimum features: %s", self.final_column_list)
            self._log_feature_transformation(
                "Safety Net",
                0,
                len(self.final_column_list),
                "All features were pruned; safety net retained a minimal set.",
            )

        # Ensure columns exist in dataframe before the final check
        self.final_column_list = [
            col for col in self.final_column_list if col in self.df.columns
        ]

        # Ensure we still have at least 1 feature
        if not self.final_column_list:
            raise NoFeaturesError(
                "CRITICAL: Unable to retain any features. The dataset might be empty or only contain the outcome variable."
            )

    def _create_xy(self):
        """Creates the feature matrix X and target vector y."""
        self.X = self.df[self.final_column_list].copy()

        # Clean up duplicated columns
        features_before = self.X.shape[1]
        self.X = clean_up_class().handle_duplicated_columns(self.X)
        if self.X.shape[1] < features_before:
            self._log_feature_transformation(
                "Drop Duplicated Columns",
                features_before,
                self.X.shape[1],
                "Removed duplicated columns.",
            )

        # This check should come first to catch non-numeric types before any processing.
        clean_up_class().screen_non_float_types(self.X)

        self.y = self.df[self.outcome_variable].copy()

        clean_up_class().handle_column_names(self.X)

        # Check for string columns or values
        if self.X.select_dtypes(include=["object", "string"]).shape[1] > 0:
            raise ValueError("DataFrame contains string (non-numeric) columns.")

        if self.X.applymap(lambda x: isinstance(x, str)).any().any():
            raise ValueError("DataFrame contains string values within numeric columns.")

        # Reset indices for clean alignment
        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)
        self._assert_index_alignment(self.X, self.y, "After initial X/y creation")

        # Check for NaNs
        if self.X.isnull().values.any():
            raise ValueError("DataFrame contains NaN values.")

        logger.info("------------------------")

    def _split_data(self):
        """Splits data into train, test, and validation sets."""
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.X_test_orig,
            self.y_test_orig,
        ) = get_data_split(X=self.X, y=self.y, local_param_dict=self.local_param_dict)

        # Reset all indices immediately after splitting
        self.X_train.reset_index(drop=True, inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)
        self.y_test.reset_index(drop=True, inplace=True)
        self.X_test_orig.reset_index(drop=True, inplace=True)
        self.y_test_orig.reset_index(drop=True, inplace=True)
        self._assert_index_alignment(self.X_train, self.y_train, "After data split")

    def _post_split_cleaning(self):
        """Applies cleaning steps post-split to prevent data leakage."""
        # Handle columns made constant by splitting
        features_before = self.X_train.shape[1]
        self.X_train, self.X_test, self.X_test_orig = (
            remove_constant_columns_with_debug(
                self.X_train, self.X_test, self.X_test_orig, verbosity=self.verbose
            )
        )
        if self.X_train.shape[1] < features_before:
            self._log_feature_transformation(
                "Drop Post-Split Constants",
                features_before,
                self.X_train.shape[1],
                "Removed columns that became constant after train/test split.",
            )

        logger.info(f"Shape of X_train after post-split cleaning: {self.X_train.shape}")

        if self.X_train.shape[1] == 0:
            raise NoFeaturesError(
                "All feature columns were removed after data splitting. Consider adjusting feature selection parameters."
            )

    def _scale_features(self):
        """Applies standard scaling to the feature sets."""
        scale = self.local_param_dict.get("scale")

        if scale:
            if self.X_train.shape[1] > 0:
                try:
                    scaler = StandardScaler()
                    self.X_train = pd.DataFrame(
                        scaler.fit_transform(self.X_train),
                        columns=self.X_train.columns,
                        index=self.X_train.index,
                    )
                    self.X_test = pd.DataFrame(
                        scaler.transform(self.X_test),
                        columns=self.X_test.columns,
                        index=self.X_test.index,
                    )
                    self.X_test_orig = pd.DataFrame(
                        scaler.transform(self.X_test_orig),
                        columns=self.X_test_orig.columns,
                        index=self.X_test_orig.index,
                    )
                    self._log_feature_transformation(
                        "Standard Scaling",
                        self.X_train.shape[1],
                        self.X_train.shape[1],
                        "Applied StandardScaler to numeric features.",
                    )
                    self._assert_index_alignment(
                        self.X_train, self.y_train, "After scaling"
                    )
                except Exception as e:
                    logger.error(f"Exception scaling data: {e}", exc_info=True)
                    logger.warning("Continuing without scaling.")
            else:
                logger.warning(
                    "Skipping scaling because no features are present in X_train."
                )

        if self.verbose >= 1:
            logger.info(
                f"len final droplist: {len(self.drop_list)} / {len(list(self.df.columns))}"
            )

    def _select_features_by_importance(self):
        """Selects features based on importance scores if configured."""
        target_n_features = self.local_param_dict.get("n_features")

        if target_n_features != "all" and self.X_train.shape[1] > 1:
            features_before = self.X_train.shape[1]

            logger.info(
                f"Shape of X_train before feature importance selection: {self.X_train.shape}"
            )

            try:
                self.X_train, self.X_test, self.X_test_orig = (
                    feature_importance_methods.handle_feature_importance_methods(
                        self,
                        target_n_features,
                        X_train=self.X_train,
                        X_test=self.X_test,
                        y_train=self.y_train,
                        X_test_orig=self.X_test_orig,
                        ml_grid_object=self,
                    )
                )
                self._log_feature_transformation(
                    "Feature Importance",
                    features_before,
                    self.X_train.shape[1],
                    "Selected top features using importance method.",
                )
                self._assert_index_alignment(
                    self.X_train, self.y_train, "After feature selection"
                )

                logger.info(
                    f"Shape of X_train after feature importance selection: {self.X_train.shape}"
                )

                if self.X_train.shape[1] == 0:
                    raise NoFeaturesError(
                        "Feature importance selection removed all features."
                    )

            except Exception as e:
                logger.error(f"Feature importance selection failed: {e}", exc_info=True)
                raise

    def _finalize_pipeline(self):
        """Final logging, checks, and model list generation."""
        if self.verbose >= 2:
            logger.info(
                f"Data Split Information:\n"
                f"Number of rows in self.X_train: {len(self.X_train)}, Columns: {self.X_train.shape[1]}\n"
                f"Number of rows in self.X_test: {len(self.X_test)}, Columns: {self.X_test.shape[1]}\n"
                f"Number of rows in self.y_train: {len(self.y_train)}\n"
                f"Number of rows in self.y_test: {len(self.y_test)}\n"
                f"Number of rows in self.X_test_orig: {len(self.X_test_orig)}, Columns: {self.X_test_orig.shape[1]}\n"
                f"Number of rows in self.y_test_orig: {len(self.y_test_orig)}",
            )

        if self.verbose >= 3:
            plot_pie_chart_with_counts(self.X_train, self.X_test, self.X_test_orig)

        # Load model class list
        self.model_class_list = [
            logisticRegressionModelGenerator(self, self.local_param_dict)
        ]

        if isinstance(self.X_train, pd.DataFrame) and self.X_train.empty:
            raise NoFeaturesError(
                "Input data X_train is an empty DataFrame. "
                "This is likely due to aggressive feature selection or data cleaning."
            )

        # Final definitive assertion
        try:
            assert_index_equal(self.X_train.index, self.y_train.index)
            logger.info(
                "Final data alignment check PASSED. X_train and y_train indices are identical."
            )
        except AssertionError:
            logger.error(
                "CRITICAL: Final data alignment check FAILED. X_train and y_train indices are NOT identical."
            )
            raise

    def _compile_and_log_feature_transformations(self, error_occurred: bool = False):
        """Compiles the feature transformation log and displays it."""
        # Ensure y_train is a pandas Series for consistency before exiting
        if hasattr(self, "y_train") and not isinstance(self.y_train, pd.Series):
            self.y_train = pd.Series(self.y_train, index=self.X_train.index)

        # Finalize the feature transformation log
        if self._feature_log_list:
            self.feature_transformation_log = pd.DataFrame(self._feature_log_list)
            log_string = self.feature_transformation_log.to_string()

            if error_occurred:
                # If an error happened, always log the transformation table for debugging
                logger.error(
                    "\n--- Feature Transformation Log (at time of error) ---\n"
                    + log_string
                )
            elif self.verbose >= 1:
                # Otherwise, log it based on verbosity
                logger.info("\n--- Feature Transformation Log ---\n" + log_string)
                logger.info("--------------------------------\n")
