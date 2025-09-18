# Hyperparameter Reference Guide

This guide provides a detailed reference for the hyperparameters defined in `ml_grid/util/grid_param_space_ga.py`. Understanding these parameters is key to customizing and optimizing your experiments.

---

## Overview

The `Grid` class in `grid_param_space_ga.py` defines the search space for the outer grid search loop. Each iteration of the experiment randomly samples a combination of these parameters.

---

### Ensemble & Learner Parameters

-   `weighted` (or `ensemble_weighting_method`): The method used to combine predictions from base learners in an ensemble.
    -   `'unweighted'`: Simple averaging of predictions. Fast and simple.
    -   `'de'`: Uses **Differential Evolution** to find optimal linear weights for the base learners. More computationally expensive but can improve performance.
    -   `'ann'`: Trains a small **Artificial Neural Network** to learn a non-linear combination of predictions. The most expensive method, but potentially the most powerful.

-   `store_base_learners`: A boolean.
    -   If `True`, every trained base learner is saved to disk. This is necessary if you want to use `use_stored_base_learners` in a subsequent run.
    -   **Warning**: This can consume a very large amount of disk space.

-   `use_stored_base_learners`: A boolean.
    -   If `True`, the experiment will load and reuse previously trained and saved base learners instead of training new ones. This can dramatically speed up experiments if you are re-running with similar data configurations.

---

### Data Preprocessing Parameters

-   `resample`: The method for handling imbalanced datasets.
    -   `'undersample'`: Randomly removes samples from the majority class.
    -   `'oversample'`: Randomly duplicates samples from the minority class.
    -   `None`: No resampling is performed.

-   `scale`: A boolean. If `True`, all features are standardized (scaled to have a mean of 0 and a standard deviation of 1). This is recommended for many algorithms (e.g., `LogisticRegression`, `SVC`).

-   `percent_missing`: A float (e.g., `99.8`). Any feature column with a percentage of missing values *greater than* this threshold will be removed from the dataset.

-   `corr`: A float (e.g., `0.9`). After one-hot encoding, if two columns have a Pearson correlation coefficient *greater than* this threshold, one of them will be removed to reduce multicollinearity.

-   `feature_selection_method`: The method used for initial feature selection.
    -   `'anova'`: Uses Analysis of Variance (F-test) to select the top features.
    -   `'markov_blanket'`: (If available) A more complex method for feature selection based on conditional independence.

---

### Genetic Algorithm Evolutionary Parameters

These parameters, defined in the `grid` dictionary, control the core evolutionary operators from the `DEAP` library.

-   `cxpb`: **Crossover Probability**. The probability that two selected parent individuals will undergo crossover to create offspring. A higher value encourages exploration of new combinations.

-   `mutpb`: **Mutation Probability**. The probability that an offspring individual will undergo mutation. Mutation is key for introducing new genetic material and preventing premature convergence.

-   `indpb`: **Individual Gene Mutation Probability**. When mutation occurs on an individual, this is the probability that each "gene" (base learner) within the individual's chromosome is swapped for a new one.

-   `t_size`: **Tournament Size**. The number of individuals randomly selected from the population to compete in a tournament. The fittest individual from the tournament is chosen as a parent for the next generation. A larger tournament size increases selection pressure.

---

### Genetic Algorithm Structural Parameters

These parameters are set in the `__init__` method of the `Grid` class and define the overall structure and duration of the GA run.

-   `nb_params`: A list of integers defining the possible sizes of the ensembles (i.e., the number of base learners in an individual's chromosome).

-   `pop_params`: A list of integers defining the possible population sizes. A larger population explores more of the search space in each generation but is more computationally expensive.

-   `g_params`: A list of integers defining the possible number of generations the algorithm will run. More generations allow for more thorough evolution but increase runtime.

---

### Miscellaneous Parameters

-   `n_features`: The number of features to use. Currently hardcoded to `['all']` in the default grid, but the framework could be extended to use numerical values for feature subset selection.

-   `param_space_size`: A string (e.g., `'small'`, `'medium'`, `'large'`) that controls the size of the hyperparameter search space for the individual base learners. This allows you to manage the trade-off between thoroughness of tuning and computational cost.

-   `n_unique_out`: The number of unique outcomes. While the project is focused on binary classification, this parameter is included for potential future extensions to multiclass problems.

-   `outcome_var_n`: A list of strings representing the suffix number of the outcome variable (e.g., `['1']` for `outcome_var_1`). This allows for future extension to multiple, differently-named outcome variables.

-   `div_p`: **Diversity Penalty**. A float (e.g., `0.1`). If set to a value greater than 0, the fitness score of an ensemble is penalized based on its lack of diversity (i.e., how similar its base learners are). This encourages the evolution of ensembles composed of different types of models, which can lead to better generalization. A value of `0` disables this feature.

---

### Feature Subset Parameters

-   `data`: This is a nested dictionary that allows you to toggle entire groups of features on or off for a given experiment run. For example, you can run an experiment that includes `'bloods'` features and another that excludes them to see their impact on performance. This is a powerful tool for high-level feature importance analysis.

By tuning these parameters, you can control everything from data preprocessing and feature engineering to the fine-grained behavior of the evolutionary search.