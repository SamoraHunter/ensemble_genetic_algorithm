from sklearn.metrics import make_scorer, roc_auc_score
from typing import Dict, Any
import logging

from ml_grid.util.config import load_config
from ml_grid.model_classes_ga.dummy_model import DummyModelGenerator
from ml_grid.model_classes_ga.adaboostClassifier_model import AdaBoostClassifierModelGenerator
from ml_grid.model_classes_ga.decisionTreeClassifier_model import DecisionTreeClassifierModelGenerator
from ml_grid.model_classes_ga.elasticNeuralNetwork_model import elasticNeuralNetworkModelGenerator
from ml_grid.model_classes_ga.extra_trees_model import extraTreesModelGenerator
from ml_grid.model_classes_ga.gaussianNB_model import GaussianNB_ModelGenerator
from ml_grid.model_classes_ga.gradientBoostingClassifier_model import GradientBoostingClassifier_ModelGenerator
from ml_grid.model_classes_ga.kNearestNeighbors_model import kNearestNeighborsModelGenerator
from ml_grid.model_classes_ga.logistic_regression_model import logisticRegressionModelGenerator
from ml_grid.model_classes_ga.mlpClassifier_model import MLPClassifier_ModelGenerator
from ml_grid.model_classes_ga.perceptron_model import perceptronModelGenerator
from ml_grid.model_classes_ga.pytorchANNBinaryClassifier_model import Pytorch_binary_class_ModelGenerator
from ml_grid.model_classes_ga.quadraticDiscriminantAnalysis_model import QuadraticDiscriminantAnalysis_ModelGenerator
from ml_grid.model_classes_ga.randomForest_model import randomForestModelGenerator
from ml_grid.model_classes_ga.svc_model import SVC_ModelGenerator
from ml_grid.model_classes_ga.XGBoost_model import XGBoostModelGenerator
logger = logging.getLogger("ensemble_ga")

class global_parameters:
    """A centralized configuration class for global project parameters.

    This class holds all the global settings that control the behavior of the
    data processing pipeline and the genetic algorithm.

    """
    
    # --- Experiment Settings ---
    input_csv_path: str
    """Path to the input dataset CSV file."""

    n_iter: int
    """The total number of grid search iterations to perform."""

    model_list: list
    """A list of model generator functions to use as base learners."""

    testing: bool
    """If True, enables testing mode with smaller datasets/parameters."""

    test_sample_n: int
    """Number of samples to use for the test set during data splitting."""

    column_sample_n: int
    """Number of columns to sample from the data. 0 means all columns."""

    # --- Execution & Logging ---
    verbose: int
    """Verbosity level for console output (0-9)."""

    debug_level: int
    """Debug level for detailed logging."""

    error_raise: bool
    """If True, raises exceptions during grid search; otherwise, logs them."""

    log_store_dataframe_path: str
    """The base name for the log file that stores experiment results."""

    # --- Model & Training ---
    knn_n_jobs: int
    """Number of parallel jobs for KNN models. -1 means using all available processors."""

    rename_cols: bool
    """If True, sanitizes column names for compatibility with libraries like XGBoost."""

    model_train_time_warning_threshold: int
    """Time in seconds after which a warning is printed for long model training times."""

    store_base_learners: bool
    """If True, saves trained base learners to disk."""

    # --- Grid Search ---
    random_grid_search: bool
    """If True, uses RandomizedSearchCV instead of GridSearchCV."""

    sub_sample_param_space_pct: float
    """The percentage of the parameter space to sample in a random grid search."""

    grid_n_jobs: int
    """Number of parallel jobs for grid search."""

    metric_list: Dict
    """A dictionary of scoring metrics for cross-validation."""

    # --- Genetic Algorithm ---
    gen_eval_score_threshold_early_stopping: int
    """The number of generations without improvement before the genetic algorithm stops early."""

    def __init__(self, config_path: str = "config.yml", **kwargs):
        """Initializes the global_parameters class.

        It follows a layered configuration approach:
        1. Hardcoded defaults.
        2. Values from a YAML config file (if provided).
        3. Runtime keyword argument overrides.

        Args:
            config_path (str, optional): Path to a custom YAML config file.
                Defaults to "config.yml".
            **kwargs: Keyword arguments to override any parameter at runtime.
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
        self.debug_level = 0

        self.knn_n_jobs = -1

        self.verbose = 3

        self.rename_cols = True

        self.error_raise = True

        self.random_grid_search = True

        self.sub_sample_param_space_pct = 0.001

        self.grid_n_jobs = 4

        self.metric_list = {
            "auc": make_scorer(roc_auc_score, needs_proba=False),
            "f1": "f1",
            "accuracy": "accuracy",
            "recall": "recall",
        }

        self.model_train_time_warning_threshold = 60

        self.store_base_learners = False

        self.gen_eval_score_threshold_early_stopping = 5

        self.log_store_dataframe_path = "log_store_dataframe"

        self.input_csv_path = "synthetic_data_for_testing.csv"

        self.n_iter = 1

        self.testing = True

        self.test_sample_n = 500

        self.column_sample_n = 30

        # Default list of model names
        default_model_names = [
            "logisticRegression", "perceptron", "extraTrees", "randomForest",
            "kNearestNeighbors", "XGBoost", "DecisionTreeClassifier",
            "AdaBoostClassifier", "elasticNeuralNetwork", "GaussianNB",
            "QuadraticDiscriminantAnalysis", "SVC", "GradientBoostingClassifier",
            "MLPClassifier", "Pytorch_binary_class"
        ]
        # Resolve model names to classes
        self.model_list = [self.MODEL_REGISTRY[name] for name in default_model_names if name in self.MODEL_REGISTRY]


        # 2. Load and merge from config file
        user_config = load_config(config_path)
        if user_config:
            global_params_config = user_config.get("global_params", {})
            for key, value in global_params_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                # Special handling for model_list to resolve strings to classes
                elif key == "model_list":
                    resolved_models = []
                    for model_name in value:
                        if model_name in self.MODEL_REGISTRY:
                            resolved_models.append(self.MODEL_REGISTRY[model_name])
                        else:
                            logger.warning("Unknown model '%s' in config file.", model_name)
                    self.model_list = resolved_models
                else:
                    logger.warning("Unknown global parameter '%s' in config file.", key)

        # 3. Apply runtime keyword argument overrides
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
