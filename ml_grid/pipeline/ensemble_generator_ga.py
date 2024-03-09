import multiprocessing
import random
import warnings
from multiprocessing import Manager, Process

import eventlet
import numpy as np
from ml_grid.pipeline.mutate_methods import baseLearnerGenerator
from ml_grid.util.global_params import global_parameters
from ml_grid.util.model_methods_ga import get_stored_model
from scipy.stats import skewnorm
from tqdm import tqdm
from functools import partial


# v3
def do_work(n=0, ml_grid_object=None):

    if ml_grid_object.verbose >= 11:
        print("do_work")

    # global_params = global_parameters.global_parameters()

    use_stored_base_learners = ml_grid_object.config_dict.get(
        "use_stored_base_learners"
    )

    # use_stored_base_learners = global_params.use_stored_base_learners

    modelFuncList = ml_grid_object.config_dict.get("modelFuncList")

    index = random.randint(0, len(modelFuncList) - 1)

    try:
        if use_stored_base_learners and random.random() > 0.5:
            if ml_grid_object.verbose >= 2:
                print("get_stored_model...")

            return get_stored_model()

        else:
            if ml_grid_object.verbose >= 11:
                print("modelFuncList[index]()")
            return modelFuncList[index](ml_grid_object, ml_grid_object.local_param_dict)
    except Exception as e:
        print(e)
        print(f"Failed to return model at index {index}, returning perceptron")
        return modelFuncList[1](ml_grid_object, ml_grid_object.local_param_dict)

    # return random.choice(modelFuncList)


# warnings.filterwarnings("RuntimeWarning")
# warnings.filterwarnings('error')
np.seterr(all="ignore")


def multi_run_wrapper(args):
    return do_work(*args)


def ensembleGenerator(nb_val=28, ml_grid_object=None):
    if ml_grid_object.verbose >= 11:
        print("Generating ensemble...ensembleGenerator", nb_val)
        print("ensembleGenerator>>ml_grid_object", ml_grid_object)

    # print("ensembleGenerator: ", nb_val)

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

    if nb_val > 1:
        #
        # if __name__ == "__main__":
        from eventlet import GreenPool

        # pool = GreenPool()
        # Create a partial function with ml_grid_object filled
        partial_do_work = partial(do_work, ml_grid_object=ml_grid_object)

        # Now use the partial function with pool.imap
        pool = multiprocessing.Pool(processes=2)
        for _ in tqdm(pool.imap(partial_do_work, dummy_list), total=len(dummy_list)):
            ensemble.append(_)

        if len(ensemble) != nb_val:

            print(
                f"Error generating ensemble {len(ensemble)} {nb_val} {len(dummy_list)}"
            )
            raise Exception(
                f"Error generating ensemble {len(ensemble)} {nb_val} {len(dummy_list)}"
            )

        if ml_grid_object.verbose >= 11:
            print("Returning ensemble of size ", len(ensemble))
            print(ensemble)
        return ensemble
    else:
        print(
            "Nb_val passed <1 Returning individual of size from baseLearnerGenerator",
            nb_val,
        )
        return baseLearnerGenerator()
