import logging
from typing import Any, Dict, List

from sklearn.metrics import make_scorer, roc_auc_score

from ml_grid.model_classes_ga.adaboostClassifier_model import (
    AdaBoostClassifierModelGenerator,
)
from ml_grid.model_classes_ga.decisionTreeClassifier_model import (
    DecisionTreeClassifierModelGenerator,
)
from ml_grid.model_classes_ga.dummy_model import DummyModelGenerator
from ml_grid.model_classes_ga.elasticNeuralNetwork_model import (
    elasticNeuralNetworkModelGenerator,
)
from ml_grid.model_classes_ga.extra_trees_model import extraTreesModelGenerator
from ml_grid.model_classes_ga.gaussianNB_model import GaussianNB_ModelGenerator
from ml_grid.model_classes_ga.gradientBoostingClassifier_model import (
    GradientBoostingClassifier_ModelGenerator,
)
from ml_grid.model_classes_ga.kNearestNeighbors_model import (
    kNearestNeighborsModelGenerator,
)
from ml_grid.model_classes_ga.logistic_regression_model import (
    logisticRegressionModelGenerator,
)
from ml_grid.model_classes_ga.mlpClassifier_model import MLPClassifier_ModelGenerator
from ml_grid.model_classes_ga.perceptron_model import perceptronModelGenerator
from ml_grid.model_classes_ga.pytorchANNBinaryClassifier_model import (
    Pytorch_binary_class_ModelGenerator,
)
from ml_grid.model_classes_ga.quadraticDiscriminantAnalysis_model import (
    QuadraticDiscriminantAnalysis_ModelGenerator,
)
from ml_grid.model_classes_ga.randomForest_model import randomForestModelGenerator
from ml_grid.model_classes_ga.svc_model import SVC_ModelGenerator
from ml_grid.model_classes_ga.XGBoost_model import XGBoostModelGenerator
from ml_grid.util.config import load_config

logger = logging.getLogger("ensemble_ga")


class global_parameters:
    """Centralized configuration class for GA and data pipeline parameters.

    This class provides a unified interface for managing all configuration
    settings used throughout the ML grid search and genetic algorithm pipeline.
    Configuration follows a three-level hierarchy that allows flexible override
    of default values:

    1. Hardcoded defaults (defined in __init__)
    2. Values from an external YAML configuration file
    3. Runtime keyword arguments passed during instantiation

    This hierarchical approach enables users to start with sensible defaults,
    customize via YAML files for repeatable experiments, and still allow
    on-the-fly overrides for dynamic scenarios.

    Attributes:
        input_csv_path (str): Path to the input dataset CSV file.
        n_iter (int): The total number of grid search iterations to perform.
        model_list (List[Any]): A list of model generator classes to use as base learners.
            Each element should be a model generator class from MODEL_REGISTRY.
        testing (bool): If True, enables testing mode with smaller datasets/parameters.
        test_sample_n (int): Number of samples to use for the test set during data splitting.
        column_sample_n (int): Number of columns to sample from the data. 0 means all columns.
        outcome_var_n (str): The identifier for the outcome variable (e.g., '1' for 'outcome_var_1').
        verbose (int): Verbosity level for console output (0-9, where 9 is most verbose).
        debug_level (int): Debug level for detailed logging (0-9, where 9 is most verbose).
        error_raise (bool): If True, raises exceptions during grid search; otherwise, logs them.
        log_store_dataframe_path (str): The base filename for the log file that stores experiment results.
        base_project_dir (str): The root directory for saving project outputs (logs, models, etc.).
        knn_n_jobs (int): Number of parallel jobs for KNN models. -1 means using all available processors.
        rename_cols (bool): If True, sanitizes column names for compatibility with libraries like XGBoost.
        model_train_time_warning_threshold (int): Time in seconds after which a warning is printed for long model training times.
        store_base_learners (bool): If True, saves trained base learners to disk.
        random_grid_search (bool): If True, uses RandomizedSearchCV instead of GridSearchCV.
        sub_sample_param_space_pct (float): The percentage of the parameter space to sample in a random grid search (0.0-1.0).
        grid_n_jobs (int): Number of parallel jobs for grid search.
        metric_list (Dict[str, Any]): A dictionary mapping metric names to scorer objects or string identifiers.
        gen_eval_score_threshold_early_stopping (int): The number of generations without improvement before the genetic algorithm stops early.

    """

    input_csv_path: str
    """Path to the input dataset CSV file."""

    n_iter: int
    """The total number of grid search iterations to perform per model generation."""

    model_list: List[Any]
    """A list of model generator classes to use as base learners.

    Each element should be a model generator class from MODEL_REGISTRY.
    Can be configured via YAML config file using string names which are
    automatically resolved to their corresponding classes during loading.
    """

    testing: bool
    """If True, enables testing mode with smaller datasets and reduced iterations."""

    test_sample_n: int
    """Number of samples to use for the test set during data splitting."""

    column_sample_n: int
    """Number of columns to sample from the data. 0 means all columns are used."""

    outcome_var_n: str
    """The identifier suffix for the outcome variable in the dataset."""

    verbose: int
    """Verbosity level for console output and logging messages."""

    debug_level: int
    """Debug level for internal debugging and tracing."""

    error_raise: bool
    """Controls exception handling during grid search execution."""

    log_store_dataframe_path: str
    """Base filename for the output CSV file storing experiment results."""

    base_project_dir: str
    """Root directory path for all project output files."""

    knn_n_jobs: int
    """Number of parallel jobs for KNN models during grid search."""

    rename_cols: bool
    """If True, sanitizes column names for compatibility with various ML libraries."""

    model_train_time_warning_threshold: int
    """Time in seconds threshold for training time warning messages."""

    store_base_learners: bool
    """If True, persists trained base learner models to disk after fitting."""

    random_grid_search: bool
    """If True, uses RandomizedSearchCV instead of exhaustive GridSearchCV."""

    sub_sample_param_space_pct: float
    """Percentage of parameter space to sample in randomized grid search."""

    grid_n_jobs: int
    """Number of parallel jobs for concurrent grid search execution."""

    metric_list: Dict[str, Any]
    """A dictionary mapping metric names to scorer objects or string identifiers."""

    gen_eval_score_threshold_early_stopping: int
    """Number of consecutive generations without score improvement before early stopping."""

    def __init__(self, config_path: str = "config.yml", **kwargs):
        """Initializes the global_parameters class with layered configuration management.

        Configuration values are resolved through a three-level hierarchy that allows
        flexible customization of default settings:

        1. Hardcoded defaults (defined in __init__ method for all parameters)
        2. Values from an external YAML configuration file specified by config_path
        3. Runtime keyword arguments passed during instantiation

        Args:
            config_path (str, optional): Path to a custom YAML config file containing
                global_params and optionally grid_params sections. Defaults to "config.yml".
            **kwargs: Arbitrary keyword arguments that override both defaults and
                configuration file values. Useful for dynamic parameter tweaking.

        Note:
            The model_list attribute has special handling throughout the configuration
            process. It can be specified as a list of model names (strings) in any
            configuration layer, which are automatically resolved to their corresponding
            model generator classes via the MODEL_REGISTRY. This allows users to
            configure models by name while ensuring type safety at runtime.
        """

        # Model registry to map string names to class constructors
        self.MODEL_REGISTRY = {
            "AdaBoostClassifier": AdaBoostClassifierModelGenerator,
            "DecisionTreeClassifier": DecisionTreeClassifierModelGenerator,
            "elasticNeuralNetwork": elasticNeuralNetworkModelGenerator,
            "extraTrees": extraTreesModelGenerator,
            "GaussianNB": GaussianNB_ModelGenerator,
            "GradientBoostingClassifier": GradientBoostingClassifier_ModelGenerator,
            "kNearestNeighbors": kNearestNeighborsModelGenerator,
            "logisticRegression": logisticRegressionModelGenerator,
            "MLPClassifier": MLPClassifier_ModelGenerator,
            "perceptron": perceptronModelGenerator,
            "Pytorch_binary_class": Pytorch_binary_class_ModelGenerator,
            "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis_ModelGenerator,
            "randomForest": randomForestModelGenerator,
            "SVC": SVC_ModelGenerator,
            "XGBoost": XGBoostModelGenerator,
            "DummyModel": DummyModelGenerator,
        }
        # 1. Set hardcoded defaults
        self.debug_level: int = 0

        self.knn_n_jobs: int = -1

        self.verbose: int = 3

        self.rename_cols: bool = True

        self.error_raise: bool = True

        self.random_grid_search: bool = True

        self.sub_sample_param_space_pct: float = 0.001

        self.grid_n_jobs: int = 4

        self.metric_list: Dict[str, Any] = {
            "auc": make_scorer(roc_auc_score, needs_proba=False),
            "f1": "f1",
            "accuracy": "accuracy",
            "recall": "recall",
        }

        self.model_train_time_warning_threshold: int = 60

        self.store_base_learners: bool = False

        self.gen_eval_score_threshold_early_stopping: int = 5

        self.log_store_dataframe_path: str = "log_store_dataframe"

        self.input_csv_path: str = "synthetic_data_for_testing.csv"

        self.n_iter: int = 1

        self.testing: bool = True

        self.test_sample_n: int = 500

        self.column_sample_n: int = 30

        self.base_project_dir: str = "HFE_GA_experiments"

        self.outcome_var_n: str = "1"

        self.max_features_to_plot: int = 60

        self.expand_plots: bool = False

        # Default list of model names
        default_model_names = [
            "logisticRegression",
            "perceptron",
            "extraTrees",
            "randomForest",
            "kNearestNeighbors",
            "XGBoost",
            "DecisionTreeClassifier",
            "AdaBoostClassifier",
            "elasticNeuralNetwork",
            "GaussianNB",
            "QuadraticDiscriminantAnalysis",
            "SVC",
            "GradientBoostingClassifier",
            "MLPClassifier",
            "Pytorch_binary_class",
        ]

        # Resolve model names to classes
        self.model_list: List[Any] = [
            self.MODEL_REGISTRY[name]
            for name in default_model_names
            if name in self.MODEL_REGISTRY
        ]

        # 2. Load and merge from config file
        user_config = load_config(config_path)
        if user_config:
            global_params_config = user_config.get("global_params", {})
            if global_params_config:  # Check if the config section is not None
                for key, value in global_params_config.items():
                    # Special handling for model_list to resolve strings to classes
                    if key == "model_list":
                        resolved_models = []
                        for model_name in value:
                            if model_name in self.MODEL_REGISTRY:
                                resolved_models.append(self.MODEL_REGISTRY[model_name])
                            else:
                                logger.warning(
                                    "Unknown model '%s' in config file.", model_name
                                )
                        if resolved_models:  # Only overwrite if we found valid models
                            self.model_list = resolved_models
                    elif hasattr(self, key):
                        setattr(self, key, value)
                    else:
                        logger.warning(
                            "Unknown global parameter '%s' in config file.", key
                        )

            # Also check grid_params for outcome_var_n which is often stored there
            grid_params_config = user_config.get("grid_params", {})
            if grid_params_config and "outcome_var_n" in grid_params_config:
                val = grid_params_config["outcome_var_n"]
                if isinstance(val, list) and len(val) > 0:
                    self.outcome_var_n = str(val[0])
                elif val is not None:
                    self.outcome_var_n = str(val)

        # 3. Apply runtime keyword argument overrides
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
