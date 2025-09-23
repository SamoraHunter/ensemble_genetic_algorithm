# Configuration Guide

This guide explains how to customize your experiments. The recommended way to configure the project is by creating a `config.yml` file in your project's root directory. This method is safe from being overwritten by package updates and keeps all your settings in one place.

---

## The `config.yml` File

The project uses a layered configuration system. By creating a `config.yml` file, you can override the default parameters.

1.  **Create the File**: Copy the `config.yml.example` from the repository root to a new file named `config.yml`.
2.  **Edit**: Uncomment and change the parameters you wish to modify. Any parameter you don't specify will use its default value.

The `config.yml` is split into three main sections:

### 1. `global_params`
These settings control the overall behavior of the experiment.
```yaml
global_params:
  verbose: 2
  grid_n_jobs: 8
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

For quick tests or dynamic settings, you can still configure the project directly in your Python scripts.
 
-   `input_csv_path`: **(Required)** The file path to your input dataset. This must be a CSV file that meets the Project Dataset Requirements.
-   `n_iter`: The number of grid search iterations to perform. Each iteration runs the genetic algorithm with a different combination of hyperparameters from the GA search space.
-   `modelFuncList`: A Python list of the model generator classes that the genetic algorithm can use as base learners. You can add or remove models from this list to control the search space. See Adding a New Base Learner.
-   `base_project_dir_global`: The root directory where all experiment results will be saved. A unique timestamped subdirectory is created here for each run.

---
## Advanced: Direct Source Code Modification (Not Recommended)
While you can still edit the default parameters in `ml_grid/util/global_params.py` and `ml_grid/util/grid_param_space_ga.py`, this is **not recommended**. Your changes will be lost when you update the package. Always prefer using a `config.yml` file.
For example, to change the default population size, you would previously have edited `grid_param_space_ga.py`:
```python
# Before
self.pop_params = [50, 100, 150]

# After
self.pop_params = [200]
```

This level of configuration gives you full control over the scope and depth of your hyperparameter search.