"""
Module providing mutation and base learner generation for the genetic algorithm.

This module implements core genetic operators used in the ensemble evolution pipeline:
1.  `baseLearnerGenerator`: Creates new base learners by randomly selecting from
    available model types defined in configuration.
2.  `mutateEnsemble`: Performs genetic mutation by replacing one base learner with
    a new random learner to maintain population diversity.

These functions work together with crossover and selection operators to drive the
evolutionary search for optimal ensemble configurations.

Module-level patterns:
    - Random selection from modelFuncList (configuration-driven)
    - Ensemble mutation preserving size while introducing diversity
    - Error handling and logging for robust GA execution

See Also:
    baseLearnerGenerator: Function for generating new base learners.
    mutateEnsemble: Function for mutating ensemble individuals.
"""

import logging
import random

logger = logging.getLogger("ensemble_ga")


def baseLearnerGenerator(ml_grid_object):
    """
    Generate a random base learner from the configuration's model function list.

    This function selects a random model function from the available model types
    defined in the ml_grid_object's configuration and instantiates it with the
    current parameter dictionary.

    Args:
        ml_grid_object: An MLGridConfigurator instance containing configuration
            data including modelFuncList (list of model constructor functions)
            and local_param_dict (parameter dictionary for model initialization).

    Returns:
        A model instance created by calling the selected model function with
        ml_grid_object and ml_grid_object.local_param_dict as arguments.

    Raises:
        IndexError: If modelFuncList is empty or None.
        KeyError: If required configuration keys are missing.
    """
    modelFuncList = ml_grid_object.config_dict.get("modelFuncList")

    index = random.randint(0, len(modelFuncList) - 1)

    return modelFuncList[index](
        ml_grid_object, ml_grid_object.local_param_dict
    )  # store as functions, pass as result of executed function


# Model will be fit in generation stage and pass fitted state with training data.


def mutateEnsemble(individual, ml_grid_object):
    """
    Mutate an ensemble individual by replacing one base learner with a new random learner.

    This function performs mutation on an ensemble individual by removing one
    existing base learner at a randomly selected position and appending a newly
    generated base learner. This maintains the ensemble size while introducing
    genetic diversity in the evolutionary pipeline.

    The mutation process:
    1. Selects a random index in the current ensemble
    2. Removes the base learner at that index (handles IndexError gracefully)
    3. Generates a new base learner using baseLearnerGenerator
    4. Appends the new learner to the ensemble

    Args:
        individual: A tuple where individual[0] is a list of base learners
            constituting the ensemble, and subsequent elements contain additional
            individual data (e.g., fitness values, metadata).
        ml_grid_object: An MLGridConfigurator instance providing configuration
            and parameter context for generating new base learners.

    Returns:
        The mutated individual with one base learner replaced by a new random learner.
        The structure of the individual is preserved with the same number of elements.

    Raises:
        Exception: Catches and re-raises any exceptions during mutation, logging
            error details including ensemble size before failure.
    """
    try:
        logger.debug("original individual of size %s:", len(individual[0]) - 1)
        if len(individual[0]) > 0:
            n = random.randint(0, len(individual[0]) - 1)
            logger.debug("Mutating individual at index %s", n)
            try:
                individual[0].pop(n)
                logger.debug("Successfully popped %s from individual", n)
            except IndexError as e:
                logger.error(
                    "Failed to pop %s from individual of length %s, popping zero",
                    n,
                    len(individual[0]),
                )
                individual[0].pop(0)

                logger.error(e)

        individual[0].append(baseLearnerGenerator(ml_grid_object))

        return individual
    except Exception as e:
        logger.error(e)
        logger.error("Failed to mutate Ensemble")
        logger.error("Len individual %s", len(individual))
        raise
