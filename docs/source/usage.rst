Usage Guide
===========

This guide explains how to use the **Ensemble Genetic Algorithm** project, including setting up your data, configuring the genetic algorithm, and running experiments.

General Workflow
----------------

To use this project, follow these general steps:

1.  **Prepare Your Data**: Ensure your input data meets the specified requirements.
2.  **Set Paths**: Define the paths to your input data in your experiment script.
3.  **Select Learning Algorithms**: Choose which base learners to include in your ensemble.
4.  **Configure Hyperparameters**: Adjust the GA's behavior and other experiment settings.
5.  **Run the Experiment**: Execute your configured genetic algorithm pipeline.

Project Dataset Requirements
----------------------------

The project expects a numeric data matrix (e.g., a Pandas DataFrame) with a binary outcome variable. The outcome variable **must** have the suffix ``_outcome_var_1``.

For more details and examples of feature column naming conventions, please refer to the synthetic data used in the unit tests or the `pat2vec` project: https://github.com/SamoraHunter/pat2vec/tree/main.


Running the Example Notebook
----------------------------

The `notebooks/example_usage.ipynb` notebook provides a complete, end-to-end example. It is designed to be executed from the command line, which is useful for remote servers or HPC clusters.

1.  **Activate the Python environment**:

    .. code-block:: bash

       source ga_env/bin/activate # Or .venv/bin/activate

2.  **Execute the notebook**:

    From the root directory of the repository, run the following command:

    .. code-block:: bash

       jupyter nbconvert --to notebook --execute notebooks/example_usage.ipynb --output notebooks/executed_example_usage.ipynb

This command runs all cells and saves the output to a new file named `executed_example_usage.ipynb`.


Configuration Guide
-------------------

This guide provides a comprehensive overview of the key configuration parameters you can adjust to customize your experiments. Configuration is split into three main areas:

1.  **Experiment Script Settings**: High-level settings in your main script (e.g., `notebooks/example_usage.ipynb`).
2.  **Data Pipeline Settings**: Parameters passed to the ``ml_grid.pipeline.data.pipe`` function.
3.  **Genetic Algorithm Hyperparameters**: The grid search space for the GA, defined in ``ml_grid/util/grid_param_space_ga.py``.

Experiment Script Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the primary parameters you will change in `notebooks/example_usage.ipynb` for each experiment.

-   ``input_csv_path``: **(Required)** The file path to your input dataset.
-   ``n_iter``: The number of grid search iterations to perform. Each iteration runs the genetic algorithm with a different combination of hyperparameters.
-   ``modelFuncList``: A Python list of the model generator classes that the genetic algorithm can use as base learners.
-   ``base_project_dir_global``: The root directory where all experiment results will be saved. A unique timestamped subdirectory is created here for each run.

Data Pipeline Settings
~~~~~~~~~~~~~~~~~~~~~~

These parameters are passed when creating the ``ml_grid_object`` inside the main experiment loop. They control data sampling and processing for each grid search iteration.

-   ``test_sample_n``: The number of samples to hold out for the final test set. These samples are not used during the GA training/validation process.
-   ``column_sample_n``: The number of feature columns to randomly sample from the dataset. If set to `0` or `None`, all columns are used.
-   ``testing``: A boolean flag. When ``True``, it uses a smaller, predefined test grid of hyperparameters, which is useful for debugging.
-   ``multiprocessing_ensemble``: A boolean flag to enable or disable multiprocessing for certain ensemble methods.

Genetic Algorithm Hyperparameter Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The search space for the genetic algorithm is defined in ``ml_grid/util/grid_param_space_ga.py``. By modifying the ``Grid`` class in this file, you can control the range of GA settings the experiment will explore.

Key parameters in the grid (``grid_param_space_ga.Grid``) include:

-   ``population_size``: A list of possible population sizes for the GA (e.g., ``[50, 100]``).
-   ``n_generations``: A list of possible values for the maximum number of generations the GA will run.
-   ``mutation_rate``: A list of probabilities for an individual (ensemble) to undergo mutation.
-   ``crossover_rate``: A list of probabilities for two individuals to perform crossover.
-   ``tournament_size``: The number of individuals to select for a tournament when choosing parents.
-   ``ensemble_weighting_method``: A list of methods to weigh the predictions of base learners. Options typically include:
    -   ``'unweighted'``: Simple averaging.
    -   ``'de'``: Differential Evolution to find optimal weights.
    -   ``'ann'``: An Artificial Neural Network to learn weights.
-   ``store_base_learners``: A boolean. If ``True``, all trained base learners are saved to disk.

How to Customize the GA Grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To change the search space, you can directly edit the lists within the ``grid_param_space_ga.py`` file. For example, to only test a population size of 200, you would change:

.. code-block:: python

   # Before
   self.pop_params = [32, 64]

   # After
   self.pop_params = [200]


Best Practices
--------------

Starting a New Experiment: Start Small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When starting with a new dataset or a new research question, avoid running a large-scale experiment immediately.

-   **Use the Test Grid**: In your experiment script (e.g., `example_usage.ipynb`), set ``testing=True`` when calling ``ml_grid.pipeline.data.pipe``. This uses a much smaller, predefined hyperparameter grid that runs very quickly, allowing you to verify that your entire pipeline works without errors.
-   **Limit Iterations**: Set ``n_iter`` to a low value (e.g., 3-5) for initial test runs. This is enough to ensure that the loop executes, data is processed, and results are saved correctly.
-   **Sample Your Data**: If your dataset is very large, consider creating a smaller, representative sample for initial exploration. This will dramatically speed up iteration cycles.

Tuning the Genetic Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After initial tests, use the results to guide the tuning of the GA itself.

-   **Check Convergence**: Examine the ``plot_all_convergence`` output. If the fitness curves are still trending upwards at the final generation, you should increase the number of generations (`g_params`). If they flatten out early, you might be able to reduce it to save computation time.
-   **Balance Population vs. Runtime**: A larger population (`pop_params`) explores the search space more thoroughly but increases runtime. Start with the defaults and only increase if you suspect the GA is failing to find good solutions due to a lack of diversity.
-   **Stick to Default Evolutionary Rates**: The crossover (`cxpb`) and mutation (`mutpb`) probabilities usually work well with default values. Only tune these if you observe specific issues like premature convergence (too little mutation) or chaotic search (too much mutation).

Managing Runtimes and Resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Large-scale experiments can be computationally expensive. Here’s how to manage them:

-   **Use Model Caching Wisely**:
    -   ``store_base_learners=True``: Set this in the grid for an initial, comprehensive run. It will save every trained base learner to disk.
    -   ``use_stored_base_learners=True``: In subsequent runs, this will load and reuse the cached models instead of retraining them, which can reduce runtime by over 90%.

-   **Be Mindful of Weighting Methods**: The ``ensemble_weighting_method`` has a major impact on runtime. `'unweighted'` is extremely fast, while `'de'` (Differential Evolution) and `'ann'` (Artificial Neural Network) are significantly more expensive.

-   **Explore Feature Subsets with ``column_sample_n``**: Use the ``column_sample_n`` parameter to randomly sample a subset of features for each experiment run. This is an efficient way to explore the feature space without creating many different data files.

Iterating on Results for Better Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The goal of the first experiment is not to find the perfect model, but to learn about the problem space.

-   **Identify Key Hyperparameters**: Use the ``plot_combined_anova_feature_importances`` and ``plot_parameter_distributions`` plots to see which hyperparameters have the biggest impact on performance.

-   **Refine Your Search Grid**: Based on the insights above, go back to ``grid_param_space_ga.py`` and narrow the search space. For example, if a `resample` value of `None` consistently performs poorly, remove it from the list.

-   **Prune Your Model List**: Check the ``plot_base_learner_feature_importance`` plot. If certain models rarely appear in top-performing ensembles, you can remove them from the `modelFuncList` to focus the search on more promising algorithms.

Final Validation is Crucial
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-   Always perform the final evaluation step using the ``EnsembleEvaluator`` on a hold-out test set.
-   The performance scores from this final step are the most realistic and unbiased measure of your model's ability to generalize to new data.