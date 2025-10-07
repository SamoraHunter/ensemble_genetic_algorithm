# Architectural Overview

This guide provides a high-level overview of the project's architecture, explaining the roles of its key components and how they interact. Understanding this structure will help you navigate, use, and extend the framework effectively.

---

## Core Components

The project is designed with a modular architecture, separating concerns into distinct components. The main workflow revolves around the following parts:

### 1. The Orchestrator (`main.py` or `example_usage.ipynb`)

This is the main entry point for running an experiment. Its primary responsibilities are:
-   **Configuration**: Loading all experiment settings from a `config.yml` file into a central `global_params` object. This includes paths, iteration counts, and model lists.
-   **Experiment Loop**: Iterating `n_iter` times to run the genetic algorithm with different hyperparameter settings.
-   **Post-Processing**: Calling the analysis and evaluation components after the main loop is complete.

### 2. The Configuration Layer (`config.yml` and `grid_param_space_ga.py`)

This layer defines the entire hyperparameter search space. The system uses a layered approach:
-   **`config.yml`**: The primary way to set parameters. You define the search space for the GA (`ga_params`) and the grid search (`grid_params`) here.
-   **`grid_param_space_ga.py`**: This file contains the default, hardcoded search space. Any values set in `config.yml` will override these defaults.

The grid includes combinations of settings for:
-   Data preprocessing (e.g., resampling, scaling).
-   Feature selection (e.g., correlation thresholds).
-   Genetic algorithm parameters (e.g., population size, mutation rate).

### 3. The Data Pipeline (`ml_grid.pipeline.data.pipe`)

This is a factory function that creates the central `ml_grid_object` for a single grid search iteration. It takes a specific set of hyperparameters from the configuration layer and performs initial data setup, including:
-   Loading the dataset.
-   Splitting data into training, validation, and hold-out test sets.
-   Applying initial data sampling or feature subset selection.

### 4. The `ml_grid_object`

This object is the heart of a single grid search iteration. It acts as a container that encapsulates everything needed for one full execution of the genetic algorithm:
-   Data splits (train, validation, test).
-   The specific hyperparameters for the current run, drawn from the `global_params` object.
-   Paths for saving logs, models, and results.
-   The list of base learner models to be used.

This object is passed to the core GA engine.

### 5. The Core GA Engine (`main_ga.py`)

This is where the evolutionary process happens. It receives the `ml_grid_object` and executes the genetic algorithm:
-   It uses the `modelFuncList` to create a population of diverse base learners.
-   It evolves ensembles (individuals) over multiple generations using selection, crossover, and mutation.
-   It evaluates the fitness of each ensemble using the validation data within the `ml_grid_object`.
-   It logs the results of the run to `final_grid_score_log.csv`.

### 6. Model Generators (`ml_grid/model_classes_ga/`)

Each file in this directory defines a "generator" for a specific machine learning model (e.g., `XGBoost`, `LogisticRegression`). A generator is a class that knows how to:
-   Define the hyperparameter search space for its model (using `hyperopt`).
-   Instantiate its model with a given set of hyperparameters.

The GA engine uses these generators to create the base learners that form the building blocks of the ensembles. This design makes the framework highly extensible. See {doc}`../adding_new_learner`.

### 7. The Analysis Layer (`GA_results_explorer`)

After all grid search iterations are complete, this class is used to analyze the results. It reads the `final_grid_score_log.csv` file and generates a suite of plots to help you understand:
-   Which hyperparameters were most impactful.
-   Which base learners performed best.
-   The convergence behavior of the GA.

See {doc}`../interpreting_results`.

### 8. The Validation Layer (`EnsembleEvaluator`)

This is the final step. The `EnsembleEvaluator` takes the best models identified during the experiment and evaluates them on the hold-out test set—data that was never seen during the entire GA process. This provides a final, unbiased measure of the models' generalization performance.

See {doc}`../evaluating_models`.

---

This modular structure allows each part of the system to be understood and modified independently, from adding a new model to changing the analysis plots.