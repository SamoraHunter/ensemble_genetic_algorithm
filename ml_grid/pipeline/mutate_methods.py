import logging
import random

logger = logging.getLogger("ensemble_ga")

def baseLearnerGenerator(ml_grid_object):

    modelFuncList = ml_grid_object.config_dict.get("modelFuncList")

    index = random.randint(0, len(modelFuncList) - 1)

    return modelFuncList[index](
        ml_grid_object, ml_grid_object.local_param_dict
    )  # store as functions, pass as result of executed function


# Model will be fit in generation stage and pass fitted state with training data.


def mutateEnsemble(individual, ml_grid_object):
    try:
        logger.debug("original individual of size %s:", len(individual[0])-1)
        n = random.randint(0, len(individual[0]) - 1)
        logger.debug("Mutating individual at index %s", n)
        try:
            individual[0].pop(n)
            logger.debug("Successfully popped %s from individual", n)
        except Exception as e:  # E722
            logger.error("Failed to pop %s from individual of length %s, popping zero", n, len(individual[0]))
            individual[0].pop(0)

            logger.error(e)

        individual[0].append(baseLearnerGenerator(ml_grid_object))

        return individual
    except Exception as e:
        logger.error(e)
        logger.error("Failed to mutate Ensemble")
        logger.error("Len individual %s", len(individual))
        raise
