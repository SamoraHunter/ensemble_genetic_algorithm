import itertools as it
import random
from typing import Dict, List, Iterator, Any
import logging

from ml_grid.util.config import load_config, merge_configs
from ml_grid.util.global_params import global_parameters
logger = logging.getLogger("ensemble_ga")


class Grid:
    """Generates and manages a grid of hyperparameter configurations for experiments.

    This class creates a Cartesian product of all specified hyperparameters,
    shuffles them, and provides an iterator to access a random sample of
    the configurations. It also defines the parameter ranges for the genetic
    algorithm itself (e.g., population size, number of generations).
    """

    test_grid: bool
    """A flag to enable a smaller, faster grid for testing purposes."""

    global_params: global_parameters
    """An instance of the global_parameters class."""

    verbose: int
    """The verbosity level, inherited from global_params."""

    sample_n: int
    """The number of hyperparameter settings to sample from the full grid."""

    grid: Dict[str, List[Any]]
    """A dictionary defining the hyperparameter search space."""

    settings_list: List[Dict[str, Any]]
    """A list of dictionaries, where each dictionary is a unique hyperparameter configuration."""

    settings_list_iterator: Iterator
    """An iterator for the `settings_list`."""

    nb_params: List[int]
    """A list of possible values for the number of base learners in an ensemble."""

    pop_params: List[int]
    """A list of possible values for the population size in the genetic algorithm."""

    g_params: List[int]
    """A list of possible values for the number of generations in the genetic algorithm."""

    def __init__(self, global_params: global_parameters, sample_n: int = 1000, test_grid: bool = False, config_path: str = "config.yml"):
        """Initializes the Grid class.

        Args:
            global_params: An initialized global_parameters object.
            sample_n: The number of hyperparameter settings to generate.
                Defaults to 1000.
            test_grid: If True, uses a smaller grid for quick testing.
                Defaults to False.
            config_path (str, optional): Path to a custom YAML config file.
                Defaults to "config.yml".
        """

        self.test_grid = test_grid

        # Use the passed-in global_params object instead of creating a new one.
        self.global_params = global_params

        self.verbose = self.global_params.verbose

        self.sample_n = sample_n if sample_n is not None else 1000

        if self.verbose >= 1:
            logger.info("Feature space slice sample_n %s", self.sample_n)

        # 1. Define default grid
        default_grid = {
            "weighted": ["ann", "de", "unweighted"],  # Weighted algorithms to consider
            # "weighted": ["unweighted"],  # An alternative option for weighted algorithms
            "use_stored_base_learners": [
                False
            ],  # Whether to reuse stored base learners
            "store_base_learners": [False],  # Per-run override for storing base learners
            "resample": ["undersample", "oversample", None],  # Resampling methods
            "scale": [True],  # Whether to scale features
            "n_features": ["all"],  # Number of features to use
            "param_space_size": ["medium"],  # Size of hyperparameter space
            "n_unique_out": [10],  # Number of unique outcomes
            "outcome_var_n": [
                "1"
            ],  # Number of outcome variable, outcome varaible should be named "outcome_var_1", "outcome_var_2", etc
            "div_p": [
                0
            ],  # Whether to use diversity weighted scoring for base learner. If >0 this metric penalised by diversity score is used to evaluate ensembles.
            "percent_missing": [
                99.9,
                99.8,
                99.7,
            ],  # Percentage of missing data column wise as a threshold to remove
            "corr": [0.9, 0.99],  # Correlation thresholds for removing columns
            # "feature_selection_method": ["markov_blanket"],
            "feature_selection_method": ["anova"],
            "cxpb": [0.5, 0.75, 0.25],  # Crossover probability
            "mutpb": [0.2, 0.4, 0.8],  # Mutation probability
            "indpb": [0.025, 0.05, 0.075],  # Probability of individual mutation
            "t_size": [3, 6, 9],  # Tournament size
            "data": [  # Data configurations
                {
                    "age": [True],  # Whether to include age
                    "sex": [True],  # Whether to include sex
                    "bmi": [True],  # Whether to include BMI
                    "ethnicity": [True],  # Whether to include ethnicity
                    "bloods": [True, False],  # Whether to include bloods
                    "diagnostic_order": [
                        True,
                        False,
                    ],  # Whether to include diagnostic orders
                    "drug_order": [True, False],  # Whether to include drug orders
                    "annotation_n": [True, False],  # Whether to include annotation_n
                    "meta_sp_annotation_n": [
                        True,
                        False,
                    ],  # Whether to include meta_sp_annotation_n
                    "annotation_mrc_n": [
                        True,
                        False,
                    ],  # Whether to include annotation_mrc_n
                    "meta_sp_annotation_mrc_n": [
                        True,
                        False,
                    ],  # Whether to include meta_sp_annotation_mrc_n
                    "core_02": [True],  # Whether to include core_02
                    "bed": [True],  # Whether to include bed
                    "vte_status": [False],  # Whether to include vte_status
                    "hosp_site": [False],  # Whether to include hosp_site
                    "core_resus": [False],  # Whether to include core_resus
                    "news": [True],  # Whether to include news
                    "date_time_stamp": [False],  # Whether to include date_time_stamp
                    "appointments": [False],  # Whether to include appointments
                }
            ],
        }

        # 2. Load and merge from config file
        self.grid = default_grid  # Always initialize with the default grid
        user_config = load_config(config_path)
        if user_config:
            grid_config = user_config.get("grid_params")
            if grid_config:  # Check if the config section is not None
                self.grid = merge_configs(default_grid, grid_config)


        def c_prod(d: Dict) -> Iterator[Dict]:
            """Recursively generates all combinations of hyperparameter settings.

            Args:
                d: A dictionary where values are lists of settings.

            Returns:
                An iterator that yields dictionaries, each representing a unique
                combination of settings.
            """

            if isinstance(d, list):
                for i in d:
                    yield from ([i] if not isinstance(i, (dict, list)) else c_prod(i))
            else:
                for i in it.product(*map(c_prod, d.values())):
                    yield dict(zip(d.keys(), i))

        self.settings_list = list(c_prod(self.grid))
        logger.info("Full settings_list size: %s", len(self.settings_list))

        random.shuffle(self.settings_list)

        self.sample_n = min(sample_n, len(self.settings_list))
        self.settings_list = random.sample(self.settings_list, self.sample_n)

        self.settings_list_iterator = iter(self.settings_list)

        # test space

        # nb_params = [4, 8, 16]
        # pop_params = [10, 20]
        # g_params = [10, 30]

        # nb_params = [4, 8, 16, 32]
        # pop_params = [32, 64, 128]
        # g_params = [128]

        # Parameters for the number of parameters for each individual in the population

        self.nb_params = [4, 8, 16, 32, 64]

        # Parameters for the population size
        self.pop_params = [32, 64, 128]

        # Parameters for the number of generations
        self.g_params = [128]

        # Override GA parameters from config file if present
        if user_config:
            ga_params_config = user_config.get("ga_params", {})
            if ga_params_config:  # Check if the config section is not None
                self.nb_params = ga_params_config.get("nb_params", self.nb_params)
                self.pop_params = ga_params_config.get("pop_params", self.pop_params)
                self.g_params = ga_params_config.get("g_params", self.g_params)

        # If test grid is enabled, override parameters for testing purposes
        if self.test_grid:
            logger.info("Testing grid enabled")
            # Override number of parameters for testing
            self.nb_params = [
                4,
                8,
            ]
            # Override population size for testing
            self.pop_params = [8]
            # Override number of generations for testing
            self.g_params = [4]
