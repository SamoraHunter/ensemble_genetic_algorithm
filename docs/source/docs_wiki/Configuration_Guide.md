# Configuration Guide

This guide provides a comprehensive overview of the key configuration parameters you can adjust to customize your experiments with the **Ensemble Genetic Algorithm** project.

---

## Overview

Configuration is split into three main areas:

1.  **Experiment Script Settings**: High-level settings in your main script (e.g., `notebooks/example_usage.ipynb`).
2.  **Data Pipeline Settings**: Parameters passed to the `ml_grid.pipeline.data.pipe` function.
3.  **Genetic Algorithm Hyperparameters**: The grid search space for the GA, defined in `ml_grid/util/grid_param_space_ga.py`.

---

## 1. Experiment Script Settings

These are the primary parameters you will change in `notebooks/example_usage.ipynb` for each experiment.

-   `input_csv_path`: **(Required)** The file path to your input dataset. This must be a CSV file that meets the Project Dataset Requirements.
-   `n_iter`: The number of grid search iterations to perform. Each iteration runs the genetic algorithm with a different combination of hyperparameters from the GA search space.
-   `modelFuncList`: A Python list of the model generator classes that the genetic algorithm can use as base learners. You can add or remove models from this list to control the search space. See Adding a New Base Learner.
-   `base_project_dir_global`: The root directory where all experiment results will be saved. A unique timestamped subdirectory is created here for each run.

---

## 2. Data Pipeline Settings

These parameters are passed when creating the `ml_grid_object` inside the main experiment loop. They control data sampling and processing for each grid search iteration.

-   `test_sample_n`: The number of samples to hold out for the final test set. These samples are not used during the GA training/validation process.
-   `column_sample_n`: The number of feature columns to randomly sample from the dataset for this specific grid search iteration. If set to `0` or `None`, all columns are used. This is a powerful way to explore different feature subsets.
-   `testing`: A boolean flag. When `True`, it uses a smaller, predefined test grid of hyperparameters, which is useful for debugging and rapid testing. Set to `False` for full-scale experiments.
-   `multiprocessing_ensemble`: A boolean flag to enable or disable multiprocessing for certain ensemble methods. Can speed up evaluation but may consume more memory.

---

## 3. Genetic Algorithm Hyperparameter Grid

The search space for the genetic algorithm's own hyperparameters is defined in `ml_grid/util/grid_param_space_ga.py`. By modifying the `Grid` class in this file, you can control the range of GA settings that the experiment will explore.

Key parameters in the grid (`grid_param_space_ga.Grid`) include:

-   `population_size`: A list of possible population sizes for the GA (e.g., `[50, 100]`). The population is the set of candidate ensembles in each generation.
-   `n_generations`: A list of possible values for the maximum number of generations the GA will run (e.g., `[50, 100]`).
-   `mutation_rate`: A list of probabilities for an individual (ensemble) to undergo mutation (e.g., `[0.1, 0.2]`).
-   `crossover_rate`: A list of probabilities for two individuals to perform crossover to create offspring (e.g., `[0.8, 0.9]`).
-   `tournament_size`: The number of individuals to select for a tournament when choosing parents for the next generation.
-   `ensemble_weighting_method`: A list of methods to weigh the predictions of base learners within an ensemble. Options typically include:
    -   `'unweighted'`: Simple averaging.
    -   `'de'`: Differential Evolution to find optimal weights.
    -   `'ann'`: An Artificial Neural Network to learn weights.
-   `store_base_learners`: A boolean. If `True`, all trained base learners are saved to disk, which can consume significant space but allows for detailed post-hoc analysis or reuse.

### How to Customize the GA Grid

To change the search space, you can directly edit the lists within the `grid_param_space_ga.py` file. For example, to only test a population size of 200, you would change:

```python
# Before
self.grid_param_space = {
    'population_size': [50, 100, 150],
    # ...
}

# After
self.grid_param_space = {
    'population_size': [200],
    # ...
}
```

This level of configuration gives you full control over the scope and depth of your hyperparameter search.