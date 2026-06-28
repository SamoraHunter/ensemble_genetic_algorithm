# Pipeline API Reference

This document provides comprehensive API reference for the data pipeline classes in the **Ensemble Genetic Algorithm** project.

---

## Table of Contents

- [Pipeline Classes](#pipeline-classes)
- [Data Processing](#data-processing)
- [Feature Selection](#feature-selection)
- [Configuration APIs](#configuration-apis)

---

## Pipeline Classes

### `ml_grid.pipeline.main_ga.run`

The primary orchestrator for running the genetic algorithm evolution process.

#### Class: `run`

Orchestrates the main Genetic Algorithm (GA) evolution process.

**Module**: `ml_grid.pipeline.main_ga`

**Instantiation Signature**:
```python
main_ga.run(
    ml_grid_object,
    local_param_dict,
    global_params
)
```

**Usage**:
```python
from ml_grid.pipeline import data, main_ga
from ml_grid.util.global_parameters import global_parameters

# Setup
global_params = global_parameters(config_path='config.yml')
ml_grid_object = data.pipe(
    global_params=global_params,
    file_name="data/dataset.csv",
    drop_term_list=[],
    local_param_dict={},
    param_space_index=0,
)

# Execute GA
result = main_ga.run(
    ml_grid_object, 
    local_param_dict={'cxpb': 0.8}, 
    global_params=global_params
).execute()

See also: [Model_API.md](Model_API.md) for model generation patterns and base learner interface details.
See also: [GA_Python_API.md](GA_Python_API.md) for genetic algorithm evaluation methods.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `global_params` | `global_parameters` | Configuration object with experiment settings |
| `ml_grid_object` | `Any` | Experiment object containing data and hyperparameters |
| `verbose` | `int` | Logging verbosity level (0-15) |
| `error_raise` | `bool` | Flag for error handling behavior |
| `nb_params` | `List[int]` | List of ensemble sizes to try |
| `pop_params` | `List[int]` | List of population sizes to try |
| `g_params` | `List[int]` | List of generation counts to try |
| `log_folder_path` | `str` | Path for storing experiment logs and artifacts |

**Methods**:

##### `execute()` → `List[List]`

Executes the full genetic algorithm process for all GA parameter combinations.

**Returns**: `List[List]`
- A list of errors encountered during execution
- Each item contains: `[model_implementation, exception, traceback]`

**Behavior**:
1. Iterates through grid of GA hyperparameters (nb, pop, g)
2. Registers genetic operators with DEAP toolbox
3. Creates initial population of candidate ensembles
4. Runs evolutionary loop (selection, crossover, mutation)
5. Tracks best-performing ensemble per configuration
6. Implements early stopping if performance stagnates
7. Evaluates final ensemble on hold-out validation set
8. Logs all results to disk

---

### `ml_grid.pipeline.main.run`

Legacy grid search orchestrator for traditional machine learning models.

#### Class: `run`

Orchestrates grid search cross-validation for predefined models (legacy module).

**Module**: `ml_grid.pipeline.main`

**Instantiation Signature**:
```python
main.run(
    ml_grid_object,
    local_param_dict
)
```

**Usage**:
```python
from ml_grid.pipeline import data, main
from ml_grid.util.global_parameters import global_parameters

# Setup
global_params = global_parameters(config_path='config.yml')
ml_grid_object = data.pipe(
    global_params=global_params,
    file_name="data/dataset.csv",
    drop_term_list=[],
    local_param_dict={},
    param_space_index=0,
)

# Execute grid search
result = main.run(
    ml_grid_object, 
    local_param_dict={'param_space_size': 'medium'}
).execute()

See also: [Model_API.md](Model_API.md) for model generation patterns.

# See Model_API documentation for model generation patterns

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `global_params` | `global_parameters` | Configuration object with experiment settings |
| `ml_grid_object` | `Any` | Experiment object containing data and hyperparameters |
| `model_class_list` | `List[Any]` | List of instantiated model classes to evaluate |
| `verbose` | `int` | Logging verbosity level |

**Methods**:

##### `execute()` → `List[List]`

Executes grid search for all configured models.

**Returns**: `List[List]`
- A list of errors encountered during execution
- Each item contains: `[model_implementation, exception, traceback]`

---

## Data Processing

### `ml_grid.pipeline.data.pipe`

Factory function for creating experiment data objects.

See also: {doc}`./Model_API` for base learner generator interface and model generation patterns.

#### Class: `pipe`

The main data processing pipeline for an ML grid experiment.

**Module**: `ml_grid.pipeline.data`

**Instantiation Signature**:
```python
data.pipe(
    global_params,
    file_name,
    drop_term_list,
    local_param_dict,
    base_project_dir,
    param_space_index,
    additional_naming=None,
    test_sample_n=0,
    column_sample_n=0,
    config_dict=None,
    testing=False,
    multiprocessing_ensemble=False
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `global_params` | `global_parameters` | Required | Global configuration object |
| `file_name` | `str` | Required | Path to input CSV file |
| `drop_term_list` | `List[str]` | Required | List of substrings for column removal |
| `local_param_dict` | `Dict` | Required | Current iteration's parameters |
| `base_project_dir` | `str` | Required | Output directory path |
| `param_space_index` | `int` | Required | Index in parameter grid |
| `additional_naming` | `str \| None` | `None` | Optional string for log folder identification |
| `test_sample_n` | `int` | 0 | Number of rows to sample (0 = all) |
| `column_sample_n` | `int` | 0 | Number of columns to sample (0 = all) |
| `config_dict` | `Dict \| None` | `None` | GA configuration options |
| `testing` | `bool` | False | Enable testing/debug mode |
| `multiprocessing_ensemble` | `bool` | False | Enable multiprocessing for ensemble |

See also: [Configuration Guide](Configuration_Guide.md) for comprehensive configuration options including parameter grids and hyperparameter search strategies.

See also: [Data_Workflow](Data_Workflow.md) for comprehensive data preprocessing details including feature scaling, correlation filtering, and train/test split strategies.

**Returns**: `ml_grid_object`

**Contents of ml_grid_object**:

- `X_train`, `y_train`: Training data splits
- `X_val`, `y_val`: Validation data (for GA fitness)
- `X_test`, `y_test`: Test data (for final evaluation)
- `model_class_list`: List of base learner generators
- `local_param_dict`: Configuration for this iteration
- `logging_paths_obj`: Paths for saving results

**Pipeline Steps**:

The pipe class performs the following steps:
1. Load data from CSV file with optional sampling
2. Select features based on configuration
3. Apply safety net if all features have been pruned
4. Create X and y variables
5. Split data into train/test/validation sets
6. Apply post-split cleaning to prevent data leakage
7. Optionally scale features using StandardScaler
8. Select features by importance if configured

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Main DataFrame holding data |
| `X_train`, `X_test`, `X_test_orig` | `pd.DataFrame` | Feature DataFrames for different splits |
| `y_train`, `y_test`, `y_test_orig` | `pd.Series` | Target Series for different splits |
| `drop_list` | `List[str]` | List of columns to remove |
| `final_column_list` | `List[str]` | Final feature column names |
| `model_class_list` | `List` | List of model generator functions |

#### Error Handling Examples

Common pipeline failures and their solutions:

```python
from ml_grid.pipeline import data
from ml_grid.util.global_parameters import global_parameters

# Scenario 1: File not found
try:
    ml_grid_object = data.pipe(
        global_params=global_parameters(config_path='config.yml'),
        file_name="data/nonexistent.csv",
        drop_term_list=[],
        local_param_dict={},
        param_space_index=0,
    )
except FileNotFoundError as e:
    print(f"Data file not found: {e}")

# Scenario 2: Invalid configuration values
try:
    ml_grid_object = data.pipe(
        global_params=global_parameters(config_path='config.yml'),
        file_name="data/dataset.csv",
        drop_term_list=[],
        local_param_dict={'invalid_param': True},
        param_space_index=0,
    )
except KeyError as e:
    print(f"Invalid parameter in local_param_dict: {e}")

# Scenario 3: Data validation errors (non-numeric columns)
try:
    ml_grid_object = data.pipe(
        global_params=global_parameters(config_path='config.yml'),
        file_name="data/dataset.csv",
        drop_term_list=[],
        local_param_dict={},
        param_space_index=0,
        testing=False,  # Validation runs in non-testing mode
    )
except ValueError as e:
    print(f"Data validation failed: {e}")

# Scenario 4: Memory errors with large datasets
import sys
try:
    ml_grid_object = data.pipe(
        global_params=global_parameters(config_path='config.yml'),
        file_name="data/large_dataset.csv",
        drop_term_list=[],
        local_param_dict={},
        param_space_index=0,
        test_sample_n=1000,  # Sample for debugging
    )
except MemoryError as e:
    print(f"Insufficient memory: {e}")
    sys.exit(1)

# Scenario 5: Multiprocessing issues on Windows
import platform
if platform.system() == 'Windows':
    importmultiprocessing.freeze_support()
```

#### Edge Cases and Limitations

**`multiprocessing_ensemble` Parameter**:

| Aspect | Details |
|--------|---------|
| **Default value** | `False` (single-process execution) |
| **Implementation** | Uses Python's `multiprocessing.Pool` for parallel model evaluation |
| **Platform compatibility** | Requires picklable objects; may fail on Windows without `if __name__ == "__main__"` guard |
| **Memory impact** | Each worker process holds a copy of data in memory |

**Known Limitations**:

1. **Process overhead**: For small ensembles (< 4 models), multiprocessing add overhead without benefit
2. **Windows compatibility**: Requires proper main block protection to avoid infinite process spawning
3. **Resource contention**: Multiple ensemble processes may compete for CPU/GPU resources
4. **Serialization limitations**: Objects with unpicklable attributes (e.g., lambda functions, file handles) will fail

**When to use `multiprocessing_ensemble=True`**:

- Large population sizes (> 128 individuals)
- Hundreds of base learners evaluated in parallel
- Long-running model training (typically > 30 seconds per model)

**When to keep `multiprocessing_ensemble=False`**:

- Small-scale experiments (< 50 models)
- Resource-constrained environments
- Debugging/development (single-process is easier to debug)

```python
# Example: Enabling multiprocessing with proper error handling
import platform

global_params = global_parameters(config_path='config.yml')

# Windows requires freeze_support for multiprocessing
if platform.system() == 'Windows':
    import multiprocessing
    multiprocessing.freeze_support()

ml_grid_object = data.pipe(
    global_params=global_params,
    file_name="data/dataset.csv",
    drop_term_list=[],
    local_param_dict={},
    param_space_index=0,
    multiprocessing_ensemble=True,  # Enable for large-scale runs
)
```

---

## Feature Selection

### `ml_grid.pipeline.get_feature_selection_class_ga.feature_selection_methods_class`

Manages feature selection methods for the pipeline.

#### Class: `feature_selection_methods_class`

Provides various feature selection techniques.

**Module**: `ml_grid.pipeline.get_feature_selection_class_ga`

**Instantiation Signature**:
```python
feature_selection_methods_class(ml_grid_object)
```

**Methods**:

##### `get_featured_selected_training_data(method="anova")` → `Tuple[pd.DataFrame, pd.DataFrame]`

Applies feature selection to training data.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"anova"` | Feature selection method |

**Returns**: `Tuple[pd.DataFrame, pd.DataFrame]`
- Selected features for training
- Selected features for testing

#### Error Handling Examples

Feature selection failures and edge cases:

```python
from ml_grid.pipeline.get_feature_selection_class_ga import feature_selection_methods_class
import pandas as pd

# Scenario 1: Insufficient data points after preprocessing
try:
    fs_method = feature_selection_methods_class(ml_grid_object)
    X_train_selected, X_test_selected = fs_method.get_featured_selected_training_data(
        method="anova"
    )
except ValueError as e:
    print(f"Insufficient samples for feature selection: {e}")
    # Fallback to all features or reduce dimensionality
    X_train_selected = ml_grid_object.X_train.copy()
    X_test_selected = ml_grid_object.X_test.copy()

# Scenario 2: Invalid method parameter
try:
    fs_method = feature_selection_methods_class(ml_grid_object)
    X_train_selected, X_test_selected = fs_method.get_featured_selected_training_data(
        method="invalid_method"
    )
except KeyError as e:
    print(f"Invalid feature selection method: {e}")
    # Use default method (anova)
    X_train_selected, X_test_selected = fs_method.get_featured_selected_training_data()

# Scenario 3: Constant features after feature selection
from sklearn.feature_selection import SelectKBest

def safe_feature_selection(ml_grid_object, method="anova"):
    """Perform feature selection with safety checks."""
    try:
        fs_method = feature_selection_methods_class(ml_grid_object)
        X_train, X_test = fs_method.get_featured_selected_training_data(method=method)
        
        # Ensure no constant features remain
        for col in X_train.columns:
            if len(X_train[col].unique()) <= 1:
                print(f"Warning: Constant feature '{col}' retained")
        
        return X_train, X_test
    except Exception as e:
        print(f"Feature selection failed, using all features: {e}")
        return ml_grid_object.X_train.copy(), ml_grid_object.X_test.copy()

# Usage with fallback
X_train_selected, X_test_selected = safe_feature_selection(ml_grid_object)

# Scenario 4: Zero variance columns after preprocessing
def remove_constant_features(df):
    """Remove constant columns to prevent feature selection errors."""
    non_const_cols = [col for col in df.columns if len(df[col].unique()) > 1]
    const_cols = [col for col in df.columns if col not in non_const_cols]
    
    if const_cols:
        print(f"Removing constant features: {const_cols}")
    
    return df[non_const_cols]

# Preprocess before feature selection
X_train_clean = remove_constant_features(ml_grid_object.X_train)
ml_grid_object.X_train = X_train_clean

fs_method = feature_selection_methods_class(ml_grid_object)
X_train_selected, X_test_selected = fs_method.get_featured_selected_training_data()
```

---

## Configuration APIs

### `ml_grid.util.config.load_config`

Loads YAML configuration files.

See also: {doc}`./Configuration_Guide` for comprehensive configuration examples and best practices.

#### Function: `load_config`

Loads a YAML configuration file from the given path.

**Module**: `ml_grid.util.config`

**Signature**:
```python
config = load_config(config_path="config.yml")
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | `str` | `"config.yml"` | Path to YAML configuration file |

**Returns**: `Dict[str, Any]`
- Configuration dictionary containing global_params and grid_params sections

**Example config.yml**:
```yaml
global_params:
  input_csv_path: "data/dataset.csv"
  n_iter: 20
  model_list: ["logisticRegression", "randomForest"]
  verbose: 3
  base_project_dir: "HFE_GA_experiments"

grid_params:
  weighted: ["unweighted"]
  resample: [null]
  corr: [0.95]

ga_params:
  nb_params: [8, 16, 24]
  pop_params: [64, 128]
  g_params: [100]
```

#### Error Handling Examples

Configuration loading failures and edge cases:

```python
from ml_grid.util.config import load_config
import os

# Scenario 1: Config file not found
try:
    config = load_config(config_path="nonexistent.yml")
except FileNotFoundError as e:
    print(f"Configuration file not found: {e}")
    # Provide default values or set explicit path
    os.environ['CONFIG_PATH'] = 'config.yml'

# Scenario 2: YAML syntax error
import yaml
try:
    config = load_config(config_path="config.yml")
except yaml.YAMLError as e:
    print(f"YAML parsing error in config file: {e}")
    # Use defaults or fix YAML syntax

# Scenario 3: Invalid parameter types
def validate_config(config):
    """Validate configuration values before use."""
    required_keys = {'global_params', 'ga_params'}
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required section: {key}")
    
    # Validate numeric parameters
    if 'n_iter' in config['global_params']:
        if not isinstance(config['global_params']['n_iter'], int):
            raise TypeError("'n_iter' must be an integer")
        if config['global_params']['n_iter'] < 1:
            raise ValueError("'n_iter' must be positive")

# Scenario 4: Missing required parameters
def load_config_with_defaults(config_path="config.yml"):
    """Load config with fallback to defaults."""
    default_config = {
        'global_params': {
            'n_iter': 20,
            'verbose': 2,
            'base_project_dir': 'experiments/',
        },
        'ga_params': {
            'nb_params': [4, 8],
            'pop_params': [50, 100],
            'g_params': [100]
        }
    }
    
    try:
        user_config = load_config(config_path)
        from ml_grid.util.config import merge_configs
        return merge_configs(default_config, user_config)
    except FileNotFoundError:
        return default_config

# Usage with error handling
try:
    config = load_config_with_defaults()
except Exception as e:
    print(f"Configuration error: {e}")

# Scenario 5: Merging nested configs with circular references
def safe_merge(default, user, depth=0, max_depth=10):
    """Safely merge configurations preventing infinite recursion."""
    if depth > max_depth:
        raise ValueError("Maximum merge depth exceeded")
    
    result = default.copy()
    for key in user:
        if isinstance(user[key], dict) and key in result:
            result[key] = safe_merge(result[key], user[key], depth + 1, max_depth)
        else:
            result[key] = user[key]
    return result

# Scenario 6: Invalid merge output (circular references detected)
try:
    config_a = {'a': 1}
    config_b = {'b': 2}
    merged = safe_merge(config_a, config_b)
except ValueError as e:
    print(f"Merge conflict: {e}")
```

---

### `ml_grid.util.config.merge_configs`

Merges user configuration into default settings.

#### Function: `merge_configs`

Recursively merges a user-defined configuration into the default one.

**Module**: `ml_grid.util.config`

**Signature**:
```python
merged_config = merge_configs(default, user)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `default` | `Dict` | Default configuration dictionary |
| `user` | `Dict` | User-defined configuration dictionary |

**Returns**: `Dict`
- Merged configuration dictionary

---

## Utility APIs

### Logging Configuration

#### Function: `ml_grid.util.logger_setup.setup_logger`

Standardized logging setup.

**Signature**:
```python
from ml_grid.util.logger_setup import setup_logger

logger = setup_logger()
```

---

## Version Information

This API documentation corresponds to version **v1.0+** of the ensemble_genetic_algorithm package.

**Python Requirement**: Python >=3.12

To check your installed version:
```bash
pip show ensemble_genetic_algorithm
```
