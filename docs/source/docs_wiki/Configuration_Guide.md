# Configuration Guide

This guide explains how to customize your experiments. The project uses a layered configuration system, which gives you flexibility in how you define settings. The order of precedence is:

1.  **Runtime Arguments** (Highest precedence): Parameters passed directly when initializing `global_parameters` in a script.
2.  **`config.yml` File**: A central YAML file in your project root for most customizations.
3.  **Hardcoded Defaults** (Lowest precedence): The default values set within the package source code.

---

## Configuration Reference

The configuration system supports three types of settings:

| Category | Parameters | Purpose |
|----------|-----------|---------|
| `global_params` | ~20 parameters | Experiment-wide settings |
| `ga_params` | 3 parameters | Genetic algorithm hyperparameters |
| `grid_params` | 30+ parameters | Data preprocessing & grid search space |

### Complete Default Configuration

```yaml
# All default values for reference
global_params:
  input_csv_path: "synthetic_sample_100_features_4.csv"
  n_iter: 20
  model_list: [
    "logisticRegression", "randomForest", "XGBoost",
    "gradientBoosting", "elasticNeuralNetwork",
    "adaboostClassifier", "decisionTreeClassifier",
    "extraTrees", "gaussianNB", "kNearestNeighbors",
    "mlpClassifier", "quadraticDiscriminantAnalysis"
  ]
  verbose: 2
  grid_n_jobs: -1
  base_project_dir: "experiments/"
  testing: false
  test_sample_n: 0
  error_raise: true

ga_params:
  nb_params: [4, 8, 16, 32, 64]
  pop_params: [32, 64, 128]
  g_params: [128]

grid_params:
  weighted: ["ann", "de", "unweighted"]
  use_stored_base_learners: [false]
  store_base_learners: [false]
  resample: ["undersample", "oversample", null]
  scale: [true]
  n_features: ["all"]
  param_space_size: ["medium"]
  n_unique_out: [10]
  outcome_var_n: ["1"]
  div_p: [0]
  percent_missing: [99.9, 99.8, 99.7]
  corr: [0.9, 0.99]
  feature_selection_method: ["anova", "markov_blanket"]
  cxpb: [0.25, 0.5, 0.75]
  mutpb: [0.2, 0.4, 0.8]
  indpb: [0.025, 0.05, 0.075]
  t_size: [3, 6, 9]
```

See {doc}`./Data_Workflow` for data preprocessing options that interact with configuration settings.

---

## The `config.yml` File

This is the **recommended method** for most configuration. It is safe from being overwritten by package updates and keeps all your settings in one place.

1.  **Create the File**: Copy the `config.yml.example` from the repository root to a new file named `config.yml`.
2.  **Edit**: Uncomment and change the parameters you wish to modify. Any parameter you don't specify will use its default value.

The `config.yml` is split into three main sections:

### 1. `global_params`

These settings control the overall behavior of the experiment, such as file paths, number of iterations, and logging verbosity.

```yaml
global_params:
  # Path to your dataset (required)
  input_csv_path: "data/my_dataset.csv"
  
  # Number of grid search iterations (higher = more thorough but slower)
  n_iter: 20
  
  # List of models for base learner pool (names from ml_grid/model_classes_ga/)
  model_list:
    - logisticRegression
    - randomForest
    - XGBoost
  
  # Verbosity level: 1=info, 2=debug, 3=pathological debug
  verbose: 2
  
  # Parallel jobs for grid search (-1 = all CPUs)
  grid_n_jobs: 8
  
  # Output directory for experiment results
  base_project_dir: "HFE_GA_experiments/"
  
  # Quick test mode (reduces all grid sizes)
  testing: false
  
  # Sample dataset rows for debugging (0=all)
  test_sample_n: 1000
  
  # Raise exceptions or continue on errors
  error_raise: true
```

#### Detailed Parameter Descriptions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_csv_path` | *required* | Path to dataset CSV file. Must end with `_outcome_var_1`. |
| `n_iter` | 20 | Number of grid search iterations (total experiment runs). |
| `model_list` | Full list | Names of model classes to use in base learner pool. |
| `verbose` | 2 | Logging level: 1=minimal, 2=standard, 3=detailed. |
| `grid_n_jobs` | -1 | Parallel jobs for CV (-1 = all cores). |
| `base_project_dir` | "experiments/" | Directory where results are saved. |
| `testing` | false | Quick test mode (reduces populations and generations). |
| `test_sample_n` | 0 | Number of rows to sample from dataset for quick testing. |

### 2. `ga_params`

These control the core genetic algorithm process.

```yaml
ga_params:
  # Number of base learners per ensemble (list = grid search over values)
  nb_params: [8, 16]
  
  # Population size - larger = more diversity but slower
  pop_params: [50, 100]
  
  # Number of generations to evolve (more = better convergence)
  g_params: [100, 200]
```

#### GA Performance Guidelines

| Population Size | Generations | Use Case |
|----------------|-------------|----------|
| 32-50 | 50-100 | Quick prototyping |
| 100-200 | 100-200 | Standard production runs |
| 200+ | 200+ | Thorough search (high compute) |

### 3. `grid_params`

This defines the hyperparameter search space for each grid search iteration.

```yaml
grid_params:
  # Weighting methods for ensemble predictions
  weighted: ["unweighted"]      # Options: "ann", "de", "unweighted"
  
  # Whether to resample data (handles class imbalance)
  resample: ["undersample", null]   # "oversample", "undersample", or null
  
  # Correlation threshold - remove highly correlated features
  corr: [0.95, 0.98]            # Higher = more aggressive filtering
  
  # Maximum % missing values allowed (above = drop column)
  percent_missing: [99]         # 99 = keep columns with ≤1% missing
  
  # Scale features to zero mean, unit variance
  scale: [true]                 # true for NN/SVM, false for trees
  
  # Number of features to retain (or "all" for all)
  n_features: ["all", 50, 100]
  
  # Feature selection method for importance scoring
  feature_selection_method: ["anova", "markov_blanket"]
```

---

## Advanced Configuration Examples

### Example 1: Medical Dataset with High Missingness

```yaml
global_params:
  input_csv_path: "data/medical_records.csv"
  n_iter: 30                                  # More iterations for complex medical data
  verbose: 2
  
  model_list:
    - logisticRegression                       # Interpretable models preferred
    - randomForest                             # Handles missing values well
    - extraTrees                               # Robust to outliers

grid_params:
  percent_missing: [95]                        # Allow some missing data
  corr: [0.90]                                 # Less aggressive filtering
  resample: ["undersample"]                    # Often imbalanced in medical data
  
ga_params:
  pop_params: [100]                            # Larger population for reliability
  g_params: [200]
```

### Example 2: Genomic Data (High Dimensionality)

```yaml
global_params:
  input_csv_path: "data/genomic_data.csv"
  
grid_params:
  scale: [true]                                # Essential for genomic features
  corr: [0.98]                                 # Keep most features
  n_features: [100, 500, "all"]               # Feature selection crucial here
  
ga_params:
  pop_params: [200]                            # Larger population for diversity
  g_params: [300]
```

### Example 3: Balanced Cybersecurity Dataset

```yaml
grid_params:
  resample: [null]                             # No need to sample if balanced
  scale: [false]                               # Often already normalized
  
ga_params:
  nb_params: [4, 8]                            # Smaller ensembles
  pop_params: [50]
  g_params: [100]                              # Faster converge possible
```

---

## Programmatic Configuration (In Scripts/Notebooks)

For quick tests or dynamic settings, you can override any parameter at runtime by passing it as a keyword argument to `global_parameters`. These arguments will take precedence over both the `config.yml` file and the hardcoded defaults.

```python
from tqdm import tqdm
from ml_grid.util.global_params import global_parameters
from ml_grid.util.grid_param_space_ga import Grid
from ml_grid.pipeline.data import pipe as data_pipe
from ml_grid.pipeline.main_ga import run as ga_run

# Load config.yml first, then override specific parameters
global_params = global_parameters(
    config_path='config.yml',
    input_csv_path="data/another_dataset.csv",  # Override from config file
    n_iter=5,                                    # Quick test instead of 20
    model_list=["logisticRegression", "XGBoost"]  # Simplified for speed
)

# Create parameter grid (loads default + merged user config)
grid = Grid(global_params=global_params, sample_n=100)

# Run experiments
for i in tqdm(range(global_params.n_iter)):
    local_param_dict = next(grid.settings_list_iterator)
    
    ml_grid_object = data_pipe(
        global_params=global_params,
        file_name=global_params.input_csv_path,
        drop_term_list=[],                       # Optional: terms to drop from features
        local_param_dict=local_param_dict,
        base_project_dir=global_params.base_project_dir,
        param_space_index=i,
    )
    
    ga_run(
        ml_grid_object=ml_grid_object,
        local_param_dict=local_param_dict,
        global_params=global_params
    ).execute()
```

### Runtime Overrides Table

| Override | Use Case |
|----------|----------|
| `verbose=3` | Debugging (detailed logs) |
| `n_iter=5` | Quick prototype |
| `model_list=[...]` | Test specific models |
| `test_sample_n=100` | Debug with subset |

This level of configuration gives you full control over the scope and depth of your hyperparameter search.

### Dynamic Configuration from Data

You can also infer parameters based on dataset characteristics:

```python
import pandas as pd

# Load a sample to infer parameters
df = pd.read_csv("data/my_data.csv", nrows=1000)

# Detect class imbalance
imbalance_ratio = df['outcome_var_1'].value_counts(normalize=True).min()

# Adjust configuration based on imbalance
if imbalance_ratio < 0.20:
    grid_params = {
        "resample": ["undersample"],
        "n_iter": 30
    }
elif imbalance_ratio < 0.40:
    grid_params = {
        "resample": ["oversample"],
        "n_iter": 25
    }
else:
    grid_params = {
        "resample": [None],
        "n_iter": 20
    }

# Use inferred parameters
global_params = global_parameters(
    config_path='config.yml',
    n_iter=grid_params["n_iter"]
)
```

See {doc}`./Implementation_Guide` for how to execute experiments with these configurations.