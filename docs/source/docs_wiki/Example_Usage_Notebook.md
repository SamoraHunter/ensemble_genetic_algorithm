0,0 +1,111 @@
# Example Usage Notebook Guide

This guide provides a detailed walkthrough of the `notebooks/example_usage.ipynb` Jupyter notebook. This notebook serves as a comprehensive, end-to-end example of how to configure and run an experiment using the **Ensemble Genetic Algorithm** project.

---

## How to Run the Notebook

The example notebook is designed to be executed from the command line, which is especially useful for running experiments on remote servers or High-Performance Computing (HPC) clusters.

1.  **Activate the Python environment**:
    Before running, ensure that all required dependencies are available by activating your project's virtual environment.

    ```bash
    source ga_env/bin/activate # Or .venv/bin/activate if you installed manually
    ```

2.  **Execute the notebook**:
    From the **root** directory of the repository, run the following command:

    ```bash
    jupyter nbconvert --to notebook --execute notebooks/example_usage.ipynb --output notebooks/executed_example_usage.ipynb
    ```

    This command runs all the cells in `example_usage.ipynb` and saves the output (including all generated plots and dataframes) to a new file named `executed_example_usage.ipynb`.

---

## Notebook Workflow

The notebook is structured to guide you through a complete machine learning experiment, from setup to final evaluation. Here is a breakdown of its workflow:

### 1. Setup and Cleanup

The first few cells prepare the environment for a new experiment.

-   **Directory Cleanup**: The notebook begins by removing the `HFE_GA_experiments` directory if it exists. This ensures that each run is clean and that results from previous experiments do not interfere with the current one.
-   **Logging Configuration**: It sets up logging to capture the experiment's progress and suppresses overly verbose output from libraries like `matplotlib`.

### 2. Experiment Configuration

The notebook is designed to be configured via the `config.yml` file in the project's root directory. Before running the notebook, you should edit `config.yml` to set key parameters like:
-   `input_csv_path`: The path to your dataset.
-   `n_iter`: The number of grid search iterations.
-   `model_list`: The base learners to use in the experiment.

### 3. Main Experiment Loop (Execution)

The core of the notebook is a `for` loop that iterates `n_iter` times (as defined in `config.yml`). In each iteration:

1.  A new set of hyperparameters is selected from the predefined search space (`grid_param_space_ga`).
2.  An `ml_grid_object` is created, which encapsulates all the settings for that specific run (data, parameters, paths, etc.).
3.  The genetic algorithm is executed via `main_ga.run(...).execute()`. This process evolves, evaluates, and saves ensembles of models.

All artifacts for the entire experiment (logs, scores, and models) are saved into a unique, timestamped subdirectory within `HFE_GA_experiments/`.

### 4. Results Analysis and Visualization

Once the experiment loop is complete, the notebook proceeds to analyze the results.

-   **Load Results**: It reads the `final_grid_score_log.csv` file, which contains performance metrics and configurations for every GA run.
-   **Initialize Explorer**: It instantiates the `GA_results_explorer` class, a powerful utility for parsing and visualizing experiment outcomes.
-   **Generate Plots**: A series of plotting functions are called to generate insightful visualizations, such as:
    -   Feature/base learner importance and co-occurrence.
    -   GA convergence curves.
    -   Performance vs. ensemble size.
    -   The impact of different hyperparameters on model performance (AUC).

These plots are saved to the experiment's results directory and provide a deep understanding of the experiment's findings.

### 5. Final Model Evaluation

The final section of the notebook performs a robust evaluation of the best-performing ensembles on unseen data.

-   It uses the `EnsembleEvaluator` class to load the best models identified during the GA search.
-   These models are then evaluated on a hold-out **test set** and **validation set**.
-   This step provides an unbiased assessment of how well the discovered ensembles generalize to new data, which is critical for validating the final models.

---

## Customizing Your Experiment

To adapt the notebook for your own research, you will primarily need to modify the `config.yml` file:

1.  **Set `input_csv_path`**: Change the file path to point to your dataset.
2.  **Adjust `n_iter`**: Increase this value for a more thorough grid search (e.g., `50` or `100`).
3.  **Modify `model_list`**: Curate the list of base learners to include the algorithms you want to explore.
4.  **Tune `ga_params` and `grid_params`**: Adjust the search space for the genetic algorithm and data processing steps as needed.

By following this structure, you can systematically run, analyze, and validate complex ensemble models for your specific classification problem.