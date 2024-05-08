from sklearn.metrics import make_scorer, roc_auc_score


class global_parameters:
    """Class that stores global parameters for the ml_grid library.

    Attributes:
        debug_level (int): Debug level, 0 is off, 1 is low, 2 is medium, 3 is high.
        knn_n_jobs (int): Number of jobs to use for knn, if -1 then use all cores.
        verbose (int): Verbosity level, 0 is silent, 1 is warnings, 2 is info, 3 more, range 0-9.
        rename_cols (bool): Whether to rename the columns of the dataframe before using it. Necessary for XGB and light GBM.
        error_raise (bool): Whether to raise an error if an issue occurs during the grid search with gridsearch and randomsearch CV.
        random_grid_search (bool): Whether to use a random search instead of exhaustive grid.
        sub_sample_param_space_pct (float): What percentage of the parameter space to sub sample for the random grid search.
        grid_n_jobs (int): Number of jobs to use for the grid search.
        metric_list (dict): Dictionary of metrics to use for the grid search, with the key being the name of the metric and the value being the metric function.
        model_train_time_warning_threshold (int): Threshold for warning if a model takes longer than this time to train (in seconds).
        store_base_learners (bool): Whether to store the base learners after training them.
        gen_eval_score_threshold_early_stopping (int): Threshold for early stopping, if the evaluation score is below the previous for this many epochs then the genetic algorithm will stop.
        log_store_dataframe_path (str): Path to store the log dataframe.
    """

    def __init__(self, debug_level=0, knn_n_jobs=-1):
        """Constructor for the global_parameters class.

        Args:
            debug_level (int): Debug level, 0 is off, 1 is low, 2 is medium, 3 is high.
            knn_n_jobs (int): Number of jobs to use for knn, if -1 then use all cores.
        """

        self.debug_level = debug_level

        self.knn_n_jobs = knn_n_jobs

        self.verbose = 3

        self.rename_cols = True

        self.error_raise = False

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
