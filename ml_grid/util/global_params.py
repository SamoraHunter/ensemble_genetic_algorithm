from sklearn.metrics import make_scorer, roc_auc_score
from typing import Dict


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

    def __init__(self, debug_level: int = 0, knn_n_jobs: int = -1):
        """Initializes the global_parameters class.

        Args:
            debug_level (int): Debug level, 0 is off, 1 is low, 2 is medium,
                3 is high. Defaults to 0.
            knn_n_jobs (int): Number of jobs to use for KNN models. -1 means
                using all available processors. Defaults to -1.
        """

        self.debug_level = debug_level

        self.knn_n_jobs = knn_n_jobs

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
