# AI Assistant Context: Ensemble Genetic Algorithm Project

## 1. Project Goal

This project uses a **Genetic Algorithm (GA)** to find the optimal **ensemble of machine learning classifiers** for binary classification tasks. It performs a grid search over a wide range of hyperparameters, including data preprocessing, feature subsets, and the GA's own parameters (e.g., population size, mutation rate).

The primary goal is to **automate the discovery of high-performing, robust ensemble models**.

---

## 2. Primary Workflow & Entry Point

The standard and recommended way to run an experiment is via the command line using `main.py`.

```bash
python main.py --config my_config.yml --evaluate --plot
```

-   **`main.py`**: The main entry point. It orchestrates the entire experiment from start to finish.
-   **`config.yml`**: The central configuration file. Users copy `config.yml.example` and modify it. This file is the primary way to control an experiment.
-   `--evaluate`: A flag to automatically evaluate the best-found model on the hold-out validation set after the main loop. This is handled by logic within `main.py` that calls `get_y_pred_resolver`.
-   `--plot`: A flag to automatically generate all analysis plots from the results using the `GA_results_explorer` class.

The `notebooks/example_usage.ipynb` file is an alternative, more interactive way to run the pipeline, primarily for development, debugging, and demonstration.

---

## 3. Key Components & Architecture

The project is modular and follows a clear pipeline structure.

### Important Concepts

-   **Configuration Hierarchy**: Settings are loaded in a layered manner, with later layers overriding earlier ones:
    1.  Hardcoded defaults in the source code (lowest precedence).
    2.  Values from the `config.yml` file.
    3.  Runtime keyword arguments passed to a class constructor (highest precedence).

-   **State Objects**:
    -   `global_params`: An object holding the configuration for the **entire experiment** (e.g., `n_iter`, file paths). It is initialized once at the start.
    -   `ml_grid_object`: An object created for **each grid search iteration**. It holds the data splits (train, validation, test) and the specific hyperparameters for a single GA run.

### Execution Flow

1.  **`main.py`**:
    -   Initializes `global_params` from `config.yml`.
    -   Initializes the `Grid` class, which creates an iterator of `n_iter` unique hyperparameter sets (`local_param_dict`).
    -   Creates a unique, timestamped directory for the experiment run.
    -   Begins a loop for `n_iter` iterations.

2.  **Inside the Loop (`main.py`)**:
    -   A `local_param_dict` is fetched from the `Grid` iterator.
    -   This dictionary is passed to `ml_grid.pipeline.data.pipe`.

3.  **`ml_grid.pipeline.data.pipe`**:
    -   This factory function reads the raw data, performs cleaning, splitting, and scaling based on the parameters.
    -   It returns a fully configured `ml_grid_object` for this specific iteration.

4.  **`ml_grid.pipeline.main_ga.run`**:
    -   The `ml_grid_object` is passed to the `run` class.
    -   The `execute()` method runs the core genetic algorithm using the `DEAP` library. It evolves a population of ensembles and logs the results for that iteration to `final_grid_score_log.csv`.

### Extensibility
-   **`ml_grid/model_classes_ga/`**: This directory contains "model generator" classes. Each class is a wrapper for a specific ML model (e.g., `XGBoost`, `logisticRegression`).
-   **Adding a New Model**: To add a new model, a developer creates a new file in this directory with a class that defines `get_hyperparameter_space()` and `get_model()`. The model's class name is then added to the `model_list` in `config.yml`.

### Analysis & Evaluation
-   **`ml_grid/util/GA_results_explorer.py`**: This class parses the final results CSV (`final_grid_score_log.csv`) and generates all analysis plots. It is called by `main.py` when the `--plot` flag is used.
-   **`ml_grid/pipeline/evaluate_methods_ga.py`**: Contains the logic for evaluating ensembles.
    -   `evaluate_weighted_ensemble_auc`: The main fitness function used by the GA during evolution. It evaluates an ensemble on the validation set.
    -   `get_y_pred_resolver`: A dispatcher that generates predictions for an ensemble based on the configured weighting method (`unweighted`, `de`, or `ann`).

---

## 4. Code Style and Conventions

This project follows standard Python best practices. For specific guidelines on coding style, testing, commit messages, and security, please refer to the detailed context file at **`.ai/guidelines.md`**.