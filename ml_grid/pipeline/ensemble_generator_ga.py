import logging
import multiprocessing
import random
from functools import partial
from typing import Any, List, Tuple

import numpy as np
from scipy.stats import skewnorm
from tqdm import tqdm

from ml_grid.pipeline.mutate_methods import baseLearnerGenerator
from ml_grid.util.model_methods_ga import get_stored_model

logger = logging.getLogger("ensemble_ga")


# v3
def do_work(n: int = 0, ml_grid_object: Any = None) -> Tuple:
    """Generates a single, trained base learner model.

    This function acts as a worker for creating one member of an ensemble. It
    randomly selects a model generator function from the `modelFuncList`.
    Based on the configuration, it can either generate a completely new model
    or retrieve a previously trained and stored model to speed up the process.

    Args:
        n: An integer argument, primarily for compatibility with multiprocessing
            pools. It is not used in the function logic. Defaults to 0.
        ml_grid_object: The main experiment object, containing configurations
            like `use_stored_base_learners` and the list of `modelFuncList`.

    Returns:
        A tuple containing the results of a single model run, typically in the
        format (mccscore, model_object, feature_list, ...).
    """
    if ml_grid_object.verbose >= 11:
        logger.debug("do_work")

    use_stored_base_learners = ml_grid_object.config_dict.get(
        "use_stored_base_learners"
    )

    modelFuncList = ml_grid_object.config_dict.get("modelFuncList")

    index = random.randint(0, len(modelFuncList) - 1)

    try:
        if use_stored_base_learners and random.random() > 0.5:
            if ml_grid_object.verbose >= 2:
                logger.debug("get_stored_model...")

            return get_stored_model(ml_grid_object)

        else:
            if ml_grid_object.verbose >= 11:
                logger.debug(
                    "Generating new model with %s", modelFuncList[index].__name__
                )
            return modelFuncList[index](ml_grid_object, ml_grid_object.local_param_dict)
    except Exception as e:
        logger.error(e)
        logger.error("Failed to return model at index %s, returning perceptron", index)
        raise e
        # Fallback to a known simple model
        return modelFuncList[1](ml_grid_object, ml_grid_object.local_param_dict)

    # return random.choice(modelFuncList)


# warnings.filterwarnings("RuntimeWarning")
# warnings.filterwarnings('error')
np.seterr(all="ignore")


def multi_run_wrapper(args: Tuple) -> Tuple:
    """A wrapper to unpack arguments for multiprocessing `do_work`.

    This function is necessary for some multiprocessing approaches that pass
    arguments as a single tuple.

    Args:
        args: A tuple of arguments to be passed to `do_work`.

    Returns:
        The result of the `do_work` function call.
    """
    return do_work(*args)


def ensembleGenerator(nb_val: int = 28, ml_grid_object: Any = None) -> List[Tuple]:
    """Generates an ensemble of a specified size by creating multiple base learners.

    This function orchestrates the creation of an entire ensemble. It first
    determines the size of the ensemble, using a skewed random distribution
    to introduce variability. It then calls the `do_work` function repeatedly
    (either sequentially or in parallel) to generate the required number of
    base learners.

    Args:
        nb_val: The maximum number of base learners for the ensemble. The actual
            number will be randomly chosen from a distribution skewed towards
            this value. Defaults to 28.
        ml_grid_object: The main experiment object, passed down to the worker
            functions.

    Returns:
        A list of tuples, where each tuple represents a trained base learner.
        This list constitutes a single ensemble (an "individual" in GA terms).
    """
    if ml_grid_object.verbose >= 11:
        logger.debug("Generating ensemble...ensembleGenerator %s", nb_val)
        logger.debug("ensembleGenerator>>ml_grid_object %s", ml_grid_object)

    # nb_val = random.randint(2, nb_val) # dynamic individual size
    max_Value = nb_val - 2
    skewness = +5
    random_val = skewnorm.rvs(
        a=skewness, loc=max_Value, size=10000
    )  # Skewnorm function

    random_val = random_val - min(
        random_val
    )  # Shift the set so the minimum value is equal to zero.
    random_val = random_val / max(
        random_val
    )  # Standadize all the vlues between 0 and 1.
    random_val = (
        random_val * max_Value
    )  # Multiply the standardized values by the maximum value.

    random_val = random_val.astype(int) + 2

    nb_val = np.random.choice(random_val)

    ensemble = []

    dummy_list = [x for x in range(0, nb_val)]

    if ml_grid_object.multiprocessing_ensemble is False and nb_val > 1:
        ensemble = []
        for _ in tqdm(dummy_list, total=len(dummy_list)):
            ensemble.append(do_work(ml_grid_object=ml_grid_object, n=_))

        if len(ensemble) != nb_val:
            logger.error(
                "Error generating ensemble %s %s %s",
                len(ensemble),
                nb_val,
                len(dummy_list),
            )
            raise Exception(
                f"Error generating ensemble {len(ensemble)} {nb_val} {len(dummy_list)}"  # f-string is fine for exceptions
            )

        if ml_grid_object.verbose >= 11:
            logger.debug("Returning ensemble of size %s", len(ensemble))
            logger.debug(ensemble)
        return ensemble

    elif nb_val > 1:
        # do multiprocessing
        # from eventlet import GreenPool

        partial_do_work = partial(do_work, ml_grid_object=ml_grid_object)
        pool = multiprocessing.Pool(processes=2)
        ensemble = []
        for _ in tqdm(pool.imap(partial_do_work, dummy_list), total=len(dummy_list)):
            ensemble.append(_)

        if len(ensemble) != nb_val:
            logger.error(
                "Error generating ensemble %s %s %s",
                len(ensemble),
                nb_val,
                len(dummy_list),
            )
            raise Exception(
                f"Error generating ensemble {len(ensemble)} {nb_val} {len(dummy_list)}"  # f-string is fine for exceptions
            )

        if ml_grid_object.verbose >= 11:
            logger.debug("Returning ensemble of size %s", len(ensemble))
            logger.debug(ensemble)
        return ensemble

    else:
        logger.warning(
            "Nb_val passed <1 Returning individual of size from baseLearnerGenerator",
            nb_val,
        )
        return baseLearnerGenerator()
