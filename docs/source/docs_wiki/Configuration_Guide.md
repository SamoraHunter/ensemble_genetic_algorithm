# Configuration Guide

This guide explains how to customize your experiments. The project uses a layered configuration system, which gives you flexibility in how you define settings. The order of precedence is:

1.  **Runtime Arguments** (Highest precedence): Parameters passed directly when initializing `global_parameters` in a script.
2.  **`config.yml` File**: A central YAML file in your project root for most customizations.
3.  **Hardcoded Defaults** (Lowest precedence): The default values set within the package source code.

---

## The `config.yml` File

This is the **recommended method** for most configuration. It is safe from being overwritten by package updates and keeps all your settings in one place.

1.  **Create the File**: Copy the `config.yml.example` from the repository root to a new file named `config.yml`.
2.  **Edit**: Uncomment and change the parameters you wish to modify. Any parameter you don't specify will use its default value.

The `config.yml` is split into three main sections:

### 1. `global_params` (in `config.yml`)
These settings control the overall behavior of the experiment, such as file paths, number of iterations, and logging verbosity.

```yaml
global_params:
  # Path to your dataset
  input_csv_path: "data/my_dataset.csv"
  # Number of grid search iterations to run
  n_iter: 20
  # List of models to include in the base learner pool
  model_list: ["logisticRegression", "randomForest", "XGBoost", "Pytorch_binary_class"]
  # Verbosity level for console output
  verbose: 2
  # Number of parallel jobs for grid search
  grid_n_jobs: 8
  # Whether to cache trained base learners to speed up subsequent runs
  store_base_learners: True
```

### 2. `ga_params`
These control the core genetic algorithm process.
```yaml
ga_params:
  nb_params: [8, 16]       # Num base learners per ensemble
  pop_params: [50]         # Population size
  g_params: [100]          # Num generations
```

### 3. `grid_params`
This defines the hyperparameter search space for each grid search iteration. You can override entire lists or specific values.
```yaml
grid_params:
  weighted: ["unweighted"] # Only use unweighted for a faster run
  resample: ["undersample", None]
  corr: [0.95]
```

---

## Programmatic Configuration (In Scripts/Notebooks)

For quick tests or dynamic settings, you can override any parameter at runtime by passing it as a keyword argument to `global_parameters`. These arguments will take precedence over both the `config.yml` file and the hardcoded defaults.

```python
from ml_grid.util.global_params import global_parameters
from ml_grid.util.grid_param_space_ga import Grid
from ml_grid.pipeline import data_pipe_grid
from ml_grid.ga_model_pipeline import main_ga

# This will load from config.yml first, then apply the overrides below
global_params = global_parameters(
    input_csv_path="data/another_dataset.csv", # Override path from config
    n_iter=5,                                  # Override n_iter for a quick run
    verbose=3                                  # Override verbosity
)

# The main loop is then executed as shown in the Quickstart section
grid = Grid(sample_n=global_params.n_iter, test_grid=global_params.testing)
for i in range(global_params.n_iter):
    local_param_dict = next(grid.settings_list_iterator)
    ml_grid_object = data_pipe_grid.pipe(global_params, local_param_dict)
    main_ga.run(ml_grid_object).execute()
```

This level of configuration gives you full control over the scope and depth of your hyperparameter search.