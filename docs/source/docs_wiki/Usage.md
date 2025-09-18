# Usage Guide

This guide explains how to use the **Ensemble Genetic Algorithm** project, including setting up your data, configuring the genetic algorithm, and running experiments.

## Project Dataset Requirements

The project expects a numeric data matrix (e.g., a Pandas DataFrame) with a binary outcome variable. The outcome variable **must** have the suffix `_outcome_var_1`.

For more details and examples of feature column naming conventions, please refer to the synthetic data used in the unit tests or the `pat2vec` project: https://github.com/SamoraHunter/pat2vec/tree/main.

## General Workflow

To use this project, follow these general steps:

1.  **Prepare Your Data**: Ensure your input data meets the specified requirements.
2.  **Set Paths**: Define the paths to your input data.
3.  **Configure Feature Space Exploration**: Decide whether to enable grid search for features.
4.  **Select Learning Algorithms**: Choose which base learners to include in your ensemble.
5.  **Configure Genetic Algorithm Hyperparameters**: Adjust the GA's behavior.
6.  **Run the Experiment**: Execute your configured genetic algorithm.

## Configuration Details

You will typically configure the project through Python scripts or Jupyter notebooks. Here are the key parameters you'll need to set:

### 1. Set Paths for Input Data

Ensure you have the necessary input data files and set their paths accordingly within your script.

### 2. Configure Feature Space Exploration

Determine which parameters or feature sets to include in the feature space exploration. This is often controlled by a `grid` dictionary or similar structure.

```python
# Example Configuration for feature space exploration
grid_config = {
    "explore_features": True,  # Set to True to enable feature space exploration
    "feature_set_1_params": {...},
    "feature_set_2_params": {...},
    # Add more parameters as needed for your specific feature exploration
}
```

### 3. Configure Learning Algorithm Inclusion

Customize the list of base learning algorithms you want to include in the ensemble. These are typically functions or class references.

```python
modelFuncList = [
    # Add learning algorithms (e.g., from scikit-learn, PyTorch, XGBoost)
    "LogisticRegression",
    "RandomForestClassifier",
    # "MyCustomPyTorchModel", # Example of a custom model
    # Add more algorithms as needed
]
```

### 4. Configure Genetic Algorithm Hyperparameters

Adjust the genetic algorithm's hyperparameters to control its search behavior:

-   `nb_params`: Maximum individual size (e.g., maximum number of base learners in an ensemble).
-   `pop_params`: Population size (number of individuals in each generation).
-   `g_params`: Maximum number of generations the algorithm will run.

```python
# Genetic algorithm hyperparameters
nb_params = 100  # Maximum individual size (e.g., max base learners in an ensemble)
pop_params = 50  # Population size
g_params = 10    # Maximum generation number
```

## Running the Example Usage Notebook

To execute and convert the example Jupyter notebook using the command line (useful for HPC environments), use the following command from the **root** of the repository:

```bash
jupyter nbconvert --to notebook --execute notebooks/example_usage.ipynb --output notebooks/executed_example_usage.ipynb
```

This command will:

-   Run the notebook `example_usage.ipynb` using the current Python environment.
-   Save the executed version as `executed_example_usage.ipynb` in the same `notebooks/` directory.
-   Preserve interactive IPython functionality (e.g., display, widgets) during execution.

📌 **Note**: Make sure the `ga_env` (or `.venv`) environment is activated before running this command:

```bash
source ga_env/bin/activate # Or .venv/bin/activate if installed manually
```
This ensures all required dependencies are available for successful execution.