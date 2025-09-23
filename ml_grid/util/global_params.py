from sklearn.metrics import make_scorer, roc_auc_score
from typing import Dict, Any

from ml_grid.util.config import load_config

class global_parameters:
    """A centralized configuration class for global project parameters.

    This class holds all the global settings that control the behavior of the
    data processing pipeline and the genetic algorithm.

    """

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

        self.store_base_learners = True

        self.gen_eval_score_threshold_early_stopping = 5

        self.log_store_dataframe_path = "log_store_dataframe"

        self.store_base_learners = True

        # 2. Load and merge from config file
        user_config = load_config(config_path)
        if user_config:
            global_params_config = user_config.get("global_params", {})
            for key, value in global_params_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"WARNING: Unknown global parameter '{key}' in config file.")

        # 3. Apply runtime keyword argument overrides
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
