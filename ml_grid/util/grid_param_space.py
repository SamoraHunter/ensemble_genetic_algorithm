import itertools as it
import logging
import random

from ml_grid.util.global_params import global_parameters

logger = logging.getLogger("ensemble_ga")


class Grid:

    def __init__(self, sample_n=1000):

        self.global_params = global_parameters()

        self.verbose = self.global_params.verbose

        if sample_n == None:
            self.sample_n = 1000
        else:
            self.sample_n = sample_n

        if self.verbose >= 1:
            logger.info("Feature space slice sample_n %s", self.sample_n)
        # Default grid
        # User can update grid dictionary on the object
        self.grid = {
            "resample": ["undersample", "oversample", None],
            "scale": [True, False],
            "feature_n": ["all"],
            "param_space_size": ["medium", "xsmall"],
            "n_unique_out": [10],
            "outcome_var_n": ["1"],
            "percent_missing": [99, 95, 80],  # n/100 ex 95 for 95% # 99.99, 99.5, 9
            "corr": [0.98, 0.99],
            "data": [
                {
                    "age": [True],
                    "sex": [True],
                    "bmi": [True],
                    "ethnicity": [True],
                    "bloods": [True],
                    "diagnostic_order": [True],
                    "drug_order": [True],
                    "annotation_n": [True],
                    "meta_sp_annotation_n": [True],
                    "annotation_mrc_n": [True],
                    "meta_sp_annotation_mrc_n": [False],
                    "core_02": [True],
                    "bed": [False],
                    "vte_status": [False],
                    "hosp_site": [False],
                    "core_resus": [True],
                    "news": [True],
                    "date_time_stamp": [False],
                }
            ],
        }

        def c_prod(d):
            if isinstance(d, list):
                for i in d:
                    yield from ([i] if not isinstance(i, (dict, list)) else c_prod(i))
            else:
                for i in it.product(*map(c_prod, d.values())):
                    yield dict(zip(d.keys(), i))

        self.settings_list = list(c_prod(self.grid))
        logger.info("Full settings_list size: %s", len(self.settings_list))

        random.shuffle(self.settings_list)

        self.settings_list = random.sample(self.settings_list, self.sample_n)

        self.settings_list_iterator = iter(self.settings_list)

        # This is likely not properly functioning. Does not return iteration, instead reinitiates.
        # Don't need to subsample, can just generate n number of random choices from grid space.
        # function can just return random choice from grid space, terminate at the other end once limit reached.
