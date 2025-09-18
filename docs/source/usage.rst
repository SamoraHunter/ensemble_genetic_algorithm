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

Configuration is split into three main areas:

1. **Experiment Script Settings**
   These are high-level settings in your main script (e.g., `notebooks/example_usage.ipynb`).

   - ``input_csv_path``: **(Required)** The file path to your input dataset.
   - ``n_iter``: The number of grid search iterations to perform.
   - ``modelFuncList``: A Python list of the model generator classes for the GA to use.

2. **Data Pipeline Settings**
   These parameters are passed to the ``ml_grid.pipeline.data.pipe`` function and control data processing for each run.

   - ``test_sample_n``: The number of samples to hold out for the final test set.
   - ``column_sample_n``: The number of feature columns to randomly sample.
   - ``testing``: Set to ``True`` to use a smaller, faster hyperparameter grid for debugging.

3. **Genetic Algorithm Hyperparameter Grid**
   The search space for the GA is defined in ``ml_grid/util/grid_param_space_ga.py``. By modifying the ``Grid`` class in this file, you can control the settings the experiment will explore.

   - ``population_size``: List of possible population sizes (e.g., ``[50, 100]``).
   - ``n_generations``: List of possible values for the number of generations.
   - ``mutation_rate``: List of mutation probabilities.
   - ``crossover_rate``: List of crossover probabilities.


Best Practices
--------------

*   **Start Small**: When beginning a new experiment, set ``testing=True`` and ``n_iter`` to a low value (e.g., 3) to quickly verify your pipeline.

*   **Check Convergence**: Examine the convergence plots. If fitness is still rising at the end, increase the number of generations (`n_generations`). If it flattens early, you may be able to reduce it.

*   **Use Model Caching**:

    -   **`store_base_learners=True`**: On an initial run, this will save all trained models to disk.
    -   **`use_stored_base_learners=True`**: In subsequent runs, this will load cached models instead of retraining them, saving significant time.

*   **Iterate on Results**: Use the analysis plots (e.g., feature importance) to understand which hyperparameters and models are most effective. Refine your search grid and model list based on these insights for future experiments.

*   **Final Validation is Crucial**: Always rely on the final evaluation on the hold-out test set as the true measure of your model's performance. The scores reported during the GA run can be optimistically biased.