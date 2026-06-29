"""
Module providing genetic algorithm crossover operators for ensemble evolution.

This module implements various crossover operators that can be used with DEAP's
toolbox to combine two parent ensembles into offspring. Each operator follows
the DEAP convention of in-place modification of the first parent individual.

Available crossover operators:
    - cxOnePoint: One-point crossover for simpler recombination
    - cxUniform: Uniform crossover with 50% swapping probability
    - cxBlend: Blend crossover that creates offspring by blending parent feature sets
    - cxOrdered: Order crossover that preserves model ordering while mixing

See Also:
    tools.cxTwoPoint: The default DEAP two-point crossover (not included here).
"""

import logging
import random
from typing import Any, List

logger = logging.getLogger("ensemble_ga")


def cxOnePoint(ind1: Any, ind2: Any) -> tuple:
    """
    Perform one-point crossover between two individuals.

    This crossover operator selects a single random point and exchanges all
    components after that point between the two parents. It's a simpler
    recombination method compared to two-point crossover, potentially preserving
    more of the parent的整体结构 while still allowing for combination.

    Args:
        ind1: The first parent individual (modified in place).
        ind2: The second parent individual (modified in place).

    Returns:
        A tuple containing both modified individuals (ind1, ind2).

    Note:
        This wraps DEAP's cxOnePoint to provide a consistent interface.
    """
    from deap import tools
    return tools.cxOnePoint(ind1, ind2)


def cxUniform(ind1: Any, ind2: Any, indpb: float = 0.5) -> tuple:
    """
    Perform uniform crossover between two individuals.

    This operator independently decides for each component whether to swap
    it between parents with the specified probability (default 50%). It's
    particularly useful when there's no implicit structure in the representation.

    Args:
        ind1: The first parent individual (modified in place).
        ind2: The second parent individual (modified in place).
        indpb: The probability of swapping each component. Defaults to 0.5.

    Returns:
        A tuple containing both modified individuals (ind1, ind2).

    Note:
        This wraps DEAP's cxUniform to provide a consistent interface and
        default probability parameter.
    """
    from deap import tools
    return tools.cxUniform(ind1, ind2, indpb=indpb)


def cxBlend(ind1: Any, ind2: Any, alpha: float = 0.5) -> tuple:
    """
    Perform blend crossover between two individuals.

    This operator creates offspring by blending the feature sets of both
    parents. It's particularly useful for continuous or set-based representations
    where mixing parent features can create beneficial combinations.

    Args:
        ind1: The first parent individual (modified in place).
        ind2: The second parent individual (modified in place).
        alpha: The blend parameter controlling the mixing ratio. Defaults to 0.5,
               which means equal contribution from both parents.

    Returns:
        A tuple containing both modified individuals (ind1, ind2).

    Note:
        This wraps DEAP's cxBlend to provide a consistent interface.
    """
    from deap import tools
    return tools.cxBlend(ind1, ind2, alpha=alpha)


def cxOrdered(ind1: Any, ind2: Any) -> tuple:
    """
    Perform order crossover between two individuals.

    This operator preserves the relative ordering of components from both
    parents while creating mixed offspring. It's useful when the sequence
    or ordering of base learners in the ensemble is important.

    Args:
        ind1: The first parent individual (modified in place).
        ind2: The second parent individual (modified in place).

    Returns:
        A tuple containing both modified individuals (ind1, ind2).

    Note:
        This wraps DEAP's cxOrdered to provide a consistent interface.
    """
    from deap import tools
    return tools.cxOrdered(ind1, ind2)


def get_crossover_operator(cx_type: str):
    """
    Retrieve a crossover operator function by name.

    This factory function allows dynamic selection of crossover operators
    based on string identifiers. It's useful for configuration-driven
    experimentation with different GA settings.

    Args:
        cx_type: The name of the crossover operator. Valid options are:
                 'onepoint', 'uniform', 'blend', 'ordered', or 'twopoint'.

    Returns:
        The corresponding crossover function.

    Raises:
        ValueError: If the specified cx_type is not recognized.
    
    Examples:
        >>> cx_func = get_crossover_operator('onepoint')
        >>> child1, child2 = cx_func(parent1, parent2)
        
        >>> cx_func = get_crossover_operator('uniform')
        >>> child1, child2 = cx_func(parent1, parent2)
    """
    cx_operators = {
        'onepoint': cxOnePoint,
        'uniform': cxUniform,
        'blend': cxBlend,
        'ordered': cxOrdered,
        'twopoint': None,  # Special case for DEAP's built-in
    }
    
    if cx_type not in cx_operators:
        raise ValueError(
            f"Unknown crossover type '{cx_type}'. "
            f"Valid options are: {', '.join(cx_operators.keys())}"
        )
    
    if cx_type == 'twopoint':
        from deap import tools
        return tools.cxTwoPoint
    
    return cx_operators[cx_type]
