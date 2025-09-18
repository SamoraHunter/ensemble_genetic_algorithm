Architectural Overview
======================

This guide provides a high-level overview of the project's architecture, explaining the roles of its key components and how they interact. Understanding this structure will help you navigate, use, and extend the framework effectively.

Core Components
---------------

The project is designed with a modular architecture, separating concerns into distinct components. The main workflow revolves around the following parts:

The Orchestrator (``example_usage.ipynb``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the main entry point for running an experiment. Its primary responsibilities are:

-   **Configuration**: Setting up high-level parameters like the input data path (``input_csv_path``), the number of grid search iterations (``n_iter``), and the list of available base learners (``modelFuncList``).
-   **Experiment Loop**: Iterating ``n_iter`` times to run the genetic algorithm with different hyperparameter settings.
-   **Post-Processing**: Calling the analysis and evaluation components after the main loop is complete.

The Configuration Layer (``grid_param_space_ga.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This component defines the entire hyperparameter search space for the experiment. It creates a "grid" of all possible combinations of settings for:

-   Data preprocessing (e.g., resampling, scaling).
-   Feature selection (e.g., correlation thresholds).
-   Genetic algorithm parameters (e.g., population size, mutation rate).

The orchestrator iterates through a random sample of these configurations.

The Data Pipeline (``ml_grid.pipeline.data.pipe``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a factory function that creates the central ``ml_grid_object`` for a single grid search iteration. It takes a specific set of hyperparameters from the configuration layer and performs initial data setup, including:

-   Loading the dataset.
-   Splitting data into training, validation, and hold-out test sets.
-   Applying initial data sampling or feature subset selection.

The ``ml_grid_object``
~~~~~~~~~~~~~~~~~~~~~~

This object is the heart of a single experiment run. It acts as a container that encapsulates everything needed for one full execution of the genetic algorithm:

-   Data splits (train, validation, test).
-   The specific hyperparameters for the current run.
-   Paths for saving logs, models, and results.
-   The list of ``modelFuncList`` to be used.

This object is passed to the core GA engine.

The Core GA Engine (``main_ga.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is where the evolutionary process happens. It receives the ``ml_grid_object`` and executes the genetic algorithm:

-   It uses the ``modelFuncList`` to create a population of diverse base learners.
-   It evolves ensembles (individuals) over multiple generations using selection, crossover, and mutation.
-   It evaluates the fitness of each ensemble using the validation data within the ``ml_grid_object``.
-   It logs the results of the run to ``final_grid_score_log.csv``.

Model Generators (``ml_grid/model_classes_ga/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each file in this directory defines a "generator" for a specific machine learning model (e.g., ``XGBoost``, ``LogisticRegression``). A generator is a class that knows how to instantiate its model with a given set of hyperparameters. The GA engine uses these generators to create the base learners that form the building blocks of the ensembles.

The Analysis Layer (``GA_results_explorer``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After all grid search iterations are complete, this class is used to analyze the results. It reads the ``final_grid_score_log.csv`` file and generates a suite of plots to help you understand which hyperparameters were most impactful, which base learners performed best, and the convergence behavior of the GA.

The Validation Layer (``EnsembleEvaluator``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the final step. The ``EnsembleEvaluator`` takes the best models identified during the experiment and evaluates them on the hold-out test set—data that was never seen during the entire GA process. This provides a final, unbiased measure of the models' generalization performance.

---

This modular structure allows each part of the system to be understood and modified independently.