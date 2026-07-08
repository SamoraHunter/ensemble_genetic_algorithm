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

# Flag to ensure runtime config is logged only once per process
_CONFIG_LOGGED = False


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

    progress_bars: bool
    """If True, shows tqdm progress bars during long operations. Default False to reduce GUI lag."""

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

        self.progress_bars: bool = False

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

        # Print config to log at initialization (only once per run)
        import ml_grid.util.global_params as gp_module

        if not gp_module._CONFIG_LOGGED:
            self._log_config()
            gp_module._CONFIG_LOGGED = True

    def _log_config(self) -> None:
        """Prints the complete configuration to the log.

        This method logs all configuration settings including defaults and
        any values loaded from the config file or passed as runtime arguments.
        """
        import datetime

        # Collect all config attributes (exclude methods, private attributes starting with _)
        config_dict = {}
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("_"):
                value = getattr(self, attr)
                if isinstance(value, (str, int, float, bool, list, dict)):
                    # Convert model classes to their names
                    if isinstance(value, list):
                        try:
                            resolved_models = []
                            for item in value:
                                item_type = type(item).__name__
                                if item_type == "function" or item_type == "type":
                                    # Try to find the model name by reversing through MODEL_REGISTRY
                                    found_name = None
                                    for name, cls in self.MODEL_REGISTRY.items():
                                        try:
                                            if item is cls:
                                                found_name = name
                                                break
                                        except Exception:
                                            pass
                                    if found_name:
                                        resolved_models.append(found_name)
                                    else:
                                        resolved_models.append(
                                            getattr(item, "__name__", str(item))
                                        )
                                else:
                                    resolved_models.append(str(item))
                            config_dict[attr] = resolved_models
                        except Exception:
                            config_dict[attr] = [str(v) for v in value]
                    elif isinstance(value, dict):
                        # For dictionary values, convert any model classes to names
                        converted_dict = {}
                        for k, v in value.items():
                            if isinstance(v, list):
                                try:
                                    resolved_models = []
                                    for item in v:
                                        item_type = type(item).__name__
                                        if (
                                            item_type == "function"
                                            or item_type == "type"
                                        ):
                                            found_name = None
                                            for (
                                                name,
                                                cls,
                                            ) in self.MODEL_REGISTRY.items():
                                                try:
                                                    if item is cls:
                                                        found_name = name
                                                        break
                                                except Exception:
                                                    pass
                                            if found_name:
                                                resolved_models.append(found_name)
                                            else:
                                                resolved_models.append(
                                                    getattr(item, "__name__", str(item))
                                                )
                                        else:
                                            resolved_models.append(str(item))
                                    converted_dict[k] = resolved_models
                                except Exception:
                                    converted_dict[k] = [str(v) for v in v]
                            elif isinstance(v, dict):
                                # Handle nested dicts (like 'data' structure)
                                converted_nested = {}
                                for nk, nv in v.items():
                                    if isinstance(nv, list):
                                        try:
                                            resolved_models = []
                                            for item in nv:
                                                item_type = type(item).__name__
                                                if (
                                                    item_type == "function"
                                                    or item_type == "type"
                                                ):
                                                    found_name = None
                                                    for (
                                                        name,
                                                        cls,
                                                    ) in self.MODEL_REGISTRY.items():
                                                        try:
                                                            if item is cls:
                                                                found_name = name
                                                                break
                                                        except Exception:
                                                            pass
                                                    if found_name:
                                                        resolved_models.append(
                                                            found_name
                                                        )
                                                    else:
                                                        resolved_models.append(
                                                            getattr(
                                                                item,
                                                                "__name__",
                                                                str(item),
                                                            )
                                                        )
                                                else:
                                                    resolved_models.append(str(item))
                                            converted_nested[nk] = resolved_models
                                        except Exception:
                                            converted_nested[nk] = [
                                                str(nv) for nv in nv
                                            ]
                                    else:
                                        converted_nested[nk] = v
                                converted_dict[k] = converted_nested
                            else:
                                converted_dict[k] = v
                        config_dict[attr] = converted_dict
                    elif hasattr(value, "__name__"):
                        # Single model class - get its name
                        model_name = None
                        for name, cls in self.MODEL_REGISTRY.items():
                            if value is cls:
                                model_name = name
                                break
                        config_dict[attr] = model_name or getattr(
                            value, "__name__", str(value)
                        )
                    else:
                        config_dict[attr] = value

        # Build pretty-printed config string
        log_lines = []
        log_lines.append("=" * 60)
        log_lines.append("RUNTIME CONFIGURATION")
        log_lines.append(f"Generated: {datetime.datetime.now().isoformat()}")
        log_lines.append("-" * 60)

        # Sort keys and group by category for better readability
        sorted_keys = sorted(config_dict.keys())
        for key in sorted_keys:
            value = config_dict[key]
            log_lines.append(f"{key}: {value}")

        log_lines.append("=" * 60)

        logger.info("\n".join(log_lines))

    @classmethod
    def reset_config_log_flag(cls) -> None:
        """Reset the config log flag, allowing configs to be logged again.

        This is useful for testing or when starting multiple independent runs
        in the same process.
        """
        import ml_grid.util.global_params as gp_module

        gp_module._CONFIG_LOGGED = False
