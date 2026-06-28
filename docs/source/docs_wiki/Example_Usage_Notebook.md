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

---

## Example: Simple Programmatic Usage

For users who prefer to integrate the GA pipeline directly into Python scripts or applications, here is a minimal example:

```python
from datetime import datetime
import os
import pathlib
from tqdm import tqdm
from ml_grid.pipeline import data, main_ga
from ml_grid.util.global_params import global_parameters
from ml_grid.util.grid_param_space_ga import Grid

# Initialize with configuration file
config_path = 'config.yml'
global_params = global_parameters(config_path=config_path)

# Create a unique experiment directory for results
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join(global_params.base_project_dir, timestamp)
pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
global_params.base_project_dir = run_dir

# Define search space
grid = Grid(
    global_params=global_params,
    config_path=config_path
)

# Main experiment loop using main_ga.run() entry point
for i in tqdm(range(global_params.n_iter)):
    local_param_dict = next(grid.settings_list_iterator)
    
    ml_grid_object = data.pipe(
        global_params=global_params,
        file_name=global_params.input_csv_path,
        local_param_dict=local_param_dict,
        param_space_index=i,
    )
    
    # GA entry point - runs genetic algorithm evolution
    main_ga.run(
        ml_grid_object, 
        local_param_dict=local_param_dict, 
        global_params=global_params
    ).execute()
```

---

## Performance Considerations

-   **Start Small**: Begin with `n_iter=3-5` to verify your pipeline works before scaling up.
-   **Use Testing Mode**: Set `testing: True` in `config.yml` for a faster, smaller grid search space.
-   **Model Caching**: For subsequent runs after the initial comprehensive experiment, set `use_stored_base_learners: True` to reuse trained models and dramatically reduce runtime.

---

## Next Steps

After completing your experiment:

1.  Review the generated plots in the output directory
2.  Analyze results using {doc}`Interpreting_Results`
3.  Validate final models on hold-out data via {doc}`Evaluating_Final_Models`
