# API Reference

This document provides a comprehensive reference for the public API of the **Ensemble Genetic Algorithm** project.

---

## Table of Contents

### Related Documentation

For additional context and guidance, refer to the following documentation:

- [Installation](Installation.md) - Set up the ensemble_genetic_algorithm package
- [Configuration Guide](Configuration_Guide.md) - Configure experiments with YAML settings
- [Pipeline API](Pipeline_API.md) - Deep dive into data and GA pipeline workflows
- [Model API](Model_API.md) - Comprehensive reference for model generators
- [GA Python API](GA_Python_API.md) - Genetic algorithm implementation details
- [Technical Deep Dive](Technical_Deep_Dive.md) - Architecture and algorithm explanation
- [Troubleshooting](Troubleshooting.md) - Common issues and solutions

---

- [Core Entry Points](#core-entry-points)
- [Configuration APIs](#configuration-apis)
- [Pipeline Classes](#pipeline-classes)
  - [Data Pipeline](#data-pipeline)
  - [GA Pipeline](#ga-pipeline)
- [Model Classes](#model-classes)
  - [Base Learner Interface](#base-learner-interface)
  - [Classification Models](#classification-models)
- [Genetic Algorithm APIs](#genetic-algorithm-apis)
  - [Evaluation Methods](#evaluation-methods)
  - [Mutation Methods](#mutation-methods)
  - [Weighting Methods](#weighting-methods)
- [Utility APIs](#utility-apis)
  - [Configuration Management](#configuration-management)
  - [Feature Selection](#feature-selection)
  - [Logging](#logging)
- [Pipeline Workflow Diagrams](#pipeline-workflow-diagrams)
- [Result Analysis APIs](#result-analysis-apis)

---

## Core Entry Points

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
from ml_grid.util.global_params import global_parameters

# Setup
global_params = global_parameters(config_path='config.yml')
ml_grid_object = data.pipe(
    global_params=global_params,
    file_name="data/dataset.csv",
    local_param_dict={},
    param_space_index=0,
)

# Execute GA
result = main_ga.run(
    ml_grid_object, 
    local_param_dict={'cxpb': 0.8}, 
    global_params=global_params
).execute()
```

See also: [Configuration Guide](Configuration_Guide.md) for comprehensive configuration options and [Pipeline_API.md](Pipeline_API.md) for pipeline workflow details.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `global_params` | `global_parameters` | Configuration object with experiment settings |
| `ml_grid_object` | `Any` | Experiment object containing data and hyperparameters |
| `verbose` | `int` | Logging verbosity level |
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

## Configuration APIs

### `ml_grid.util.global_params`

Central configuration object for experiments.

#### Class: `global_parameters`

Controls overall experiment behavior and settings.

**Module**: `ml_grid.util.global_params`

**Instantiation Signature**:
```python
global_parameters(
    config_path=None,
    **kwargs
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | `str \| None` | `None` | Path to YAML configuration file |
| `\*\*kwargs` | any | - | Runtime parameter overrides |

**Available Parameters** (from config file):

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `input_csv_path` | `str` | `"data/my.csv"` | Path to input dataset |
| `n_iter` | `int` | 20 | Number of grid search iterations |
| `model_list` | `List[str]` | `["logisticRegression"]` | Base learners to use |
| `verbose` | `int` | 2 | Logging verbosity (0-15) |
| `grid_n_jobs` | `int` | 8 | Parallel jobs for grid search |
| `base_project_dir` | `str` | `"HFE_GA_experiments"` | Output directory |
| `testing` | `bool` | False | Use smaller test grid |

---

### `ml_grid.util.grid_param_space_ga.Grid`

Defines the hyperparameter search space for experiments.

#### Class: `Grid`

Creates parameter grids for systematic exploration of the configuration space.

**Module**: `ml_grid.util.grid_param_space_ga`

**Instantiation Signature**:
```python
Grid(
    global_params,
    config_path=None,
    test_grid=False
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `global_params` | `global_parameters` | Required | Experiment configuration |
| `config_path` | `str \| None` | `None` | Override config file path |
| `test_grid` | `bool` | `False` | Use smaller test grid |

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `settings_list_iterator` | `Iterator[Dict]` | Yields parameter combinations |
| `nb_params` | `List[int]` | Ensemble sizes: `[8, 16, 24]` |
| `pop_params` | `List[int]` | Population sizes: `[64, 128]` |
| `g_params` | `List[int]` | Generation counts: `[100]` |

**Usage**:
```python
global_params = global_parameters(config_path='config.yml')
grid = Grid(global_params=global_params)

# Iterate through parameter combinations
for i in range(20):
    params = next(grid.settings_list_iterator)
    # params contains: weighted, resample, corr, etc.
```

---

## Pipeline Classes

### Data Pipeline

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

**Returns**: `ml_grid_object`

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
| `feature_transformation_log` | `pd.DataFrame` | Log of feature counts at each pipeline step |

**Edge Cases and Special Behavior**:

##### Handling Empty Feature Sets

The data pipeline includes safety mechanisms to prevent failures when all features are pruned during data cleaning:

1. **Safety Net**: If the feature selection process removes all features (e.g., due to high correlation `corr` threshold, missing value threshold `percent_missing`, or constant column removal), the pipeline activates a safety net that retains at least 1-2 numeric non-outcome columns from the original dataset.

2. **NoFeaturesError Exception**: If the safety net cannot retain any features (e.g., the dataset is empty or contains only the outcome variable), a `NoFeaturesError` exception is raised with a clear message indicating the root cause.

##### Empty Selection via max_features Parameter

When `n_features` parameter in `local_param_dict` results in zero features selection:

- **Scenario**: The feature importance method (ANOVA F-test or Markov Blanket) selects fewer features than requested, possibly resulting in an empty set if no features pass statistical significance tests.

- **Behavior**: 
  - If the selected feature count reaches zero during `_select_features_by_importance()`, a `NoFeaturesError` is raised with message: `"Feature importance selection removed all features."`
  - This occurs after post-split cleaning, which may eliminate columns that became constant

##### Handling Edge Cases Programmatically

Example of how to handle empty feature scenarios:

```python
from ml_grid.pipeline import data, main_ga
from ml_grid.util.global_params import global_parameters
from ml_grid.pipeline.data import NoFeaturesError

try:
    # Setup with aggressive feature selection
    global_params = global_parameters(config_path='config.yml')
    
    # Use a safe configuration with fallback behavior
    local_param_dict = {
        'corr': 0.95,              # Moderate correlation threshold
        'percent_missing': 100.0,   # Allow some missing values (90-100%)
        'n_features': 3,            # Request 3 features, but may get fewer
        'feature_selection_method': 'anova'  # or 'markov_blanket'
    }
    
    ml_grid_object = data.pipe(
        global_params=global_params,
        file_name="data/dataset.csv",
        drop_term_list=[],
        local_param_dict=local_param_dict,
        param_space_index=0,
    )
    
except NoFeaturesError as e:
    # Handle the edge case gracefully
    logger.warning(f"Feature selection failed: {e}")
    
    # Fallback strategies:
    # 1. Relax feature selection parameters
    relaxed_params = {
        'n_features': 'all',         # Use all remaining features
        'corr': 0.85,                # Lower correlation threshold
        'percent_missing': 99.0      # More lenient missing value threshold
    }
    
    # 2. Or disable feature importance selection temporarily
    safe_params = {
        'n_features': 'all',         # Explicitly request all
        'feature_selection_method': None
    }
    
except ValueError as e:
    # Handle data type errors (non-numeric columns, string values)
    logger.error(f"Data validation error: {e}")
```

**Best Practices for Robust Experiments**:

1. **Use Safety Net**: Always enable the default safety net by ensuring `n_features != 'all'` only when you have sufficient features.
2. **Monitor Feature Logs**: Check the `feature_transformation_log` attribute after pipeline execution to track feature counts at each step.
3. **Gradual Aggression**: Start with lenient thresholds (higher `percent_missing`, lower `corr`) and gradually increase aggressiveness.
4. **Validate Input Data**: Ensure your dataset has sufficient numeric features beyond the outcome variable before running experiments.

##### Accessing Feature Transformation Log

After pipeline execution, examine which features were removed at each step:

```python
# After creating ml_grid_object
print(ml_grid_object.feature_transformation_log)

# Example output:
#          step  features_before  features_after  features_changed        description
# 0   Initial Load             50              50                 0     Initial data loaded.
# 1 Feature Selection           50              48                -2  Selected columns based on feature toggles
# 2    Drop Correlated           48              45                -3  Dropped columns with correlation > 0.95
# 3      Drop Missing            45              45                 0  Dropped columns with > 99% missing
# 4   Drop Other Outcomes        45              45                 0  Removed other potential outcome variables
# 5     Drop Constants           45              38                -7         Removed constant columns
```

This log helps diagnose why features were removed and enables better parameter tuning for future runs.

---

### GA Pipeline

#### Class: `run`

The primary orchestrator for running the genetic algorithm evolution process.

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
```

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

## Model Classes

### Base Learner Interface

All base learner generators follow a consistent interface pattern (see [Model_API.md](Model_API.md) for comprehensive model generator reference).

**Generation Function Signature**:
```python
def model_nameModelGenerator(
    ml_grid_object: Any,
    local_param_dict: Dict
) -> Tuple[float, ModelClass, List[str], int, float, np.ndarray]:
    """Generates, trains, and evaluates a model.
    
    Args:
        ml_grid_object: Contains X_train, y_train, X_test, y_test and config
        local_param_dict: Parameters for this specific run
        
    Returns:
        Tuple of (mccscore, model, feature_names, train_time, auc_score, y_pred)
    """
```

**Return Values**:

| Index | Type | Description |
|-------|------|-------------|
| 0 | `float` | Matthews Correlation Coefficient (MCC) |
| 1 | `ModelClass` | Trained model object |
| 2 | `List[str]` | List of feature names used for training |
| 3 | `int` | Model training time in seconds |
| 4 | `float` | ROC AUC score |
| 5 | `np.ndarray` | Model predictions on test set |

**Available Models**:
- `AdaBoostClassifierModelGenerator`
- `DecisionTreeClassifierModelGenerator`
- `elasticNeuralNetworkModelGenerator`
- `extraTreesModelGenerator`
- `GaussianNB_ModelGenerator`
- `GradientBoostingClassifier_ModelGenerator`
- `kNearestNeighborsModelGenerator`
- `logisticRegressionModelGenerator`
- `MLPClassifier_ModelGenerator`
- `perceptronModelGenerator`
- `Pytorch_binary_class_ModelGenerator`
- `QuadraticDiscriminantAnalysis_ModelGenerator`
- `randomForestModelGenerator`
- `SVC_ModelGenerator`
- `XGBoostModelGenerator`

---

### Classification Models

#### Function: `logisticRegressionModelGenerator`

Generates, trains, and evaluates a logistic regression classifier.

**Module**: `ml_grid.model_classes_ga.logistic_regression_model`

**Generation Signature**:
```python
lr_generator = logisticRegressionModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

#### Function: `randomForestModelGenerator`

Generates, trains, and evaluates a random forest classifier.

**Module**: `ml_grid.model_classes_ga.randomForest_model`

**Generation Signature**:
```python
rf_generator = randomForestModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

#### Function: `XGBoostModelGenerator`

Generates, trains, and evaluates an XGBoost classifier.

**Module**: `ml_grid.model_classes_ga.XGBoost_model`

**Generation Signature**:
```python
xgb_generator = XGBoostModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

---

## Genetic Algorithm APIs

### Evaluation Methods

#### Function: `get_y_pred_resolver`

Resolves and generates predictions for ensemble evaluation.

**Module**: `ml_grid.pipeline.evaluate_methods_ga`

**Signature**:
```python
y_pred = get_y_pred_resolver(
    individual,
    ml_grid_object,
    valid=False
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `individual` | `List` | Required | Ensemble configuration (DEAP个体格式) |
| `ml_grid_object` | `Any` | Required | Experiment object with data splits |
| `valid` | `bool` | `False` | If True, predict on validation set |

**Returns**: `Union[List, np.ndarray]`
- Final ensemble predictions

#### Function: `evaluate_weighted_ensemble_auc`

Main fitness evaluation function for genetic algorithm.

**Module**: `ml_grid.pipeline.evaluate_methods_ga`

**Signature**:
```python
fitness = evaluate_weighted_ensemble_auc(
    individual,
    ml_grid_object
)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `individual` | `List` | Ensemble to evaluate |
| `ml_grid_object` | `Any` | Experiment object with data and configuration |

**Returns**: `Tuple[float]`
- Single-element tuple containing fitness score (AUC or diversity-penalized AUC)

---

### Mutation Methods

#### Function: `baseLearnerGenerator`

Generates a random base learner.

**Module**: `ml_grid.pipeline.mutate_methods`

**Signature**:
```python
new_learner = baseLearnerGenerator(ml_grid_object)
```

#### Function: `mutateEnsemble`

Mutates an ensemble by replacing one base learner.

**Module**: `ml_grid.pipeline.mutate_methods`

**Signature**:
```python
mutated_individual = mutateEnsemble(
    individual,
    ml_grid_object
)
```

---

### Weighting Methods

#### Function: `get_unweighted_ensemble_predictions`

Generates predictions by majority voting (mode).

**Module**: `ml_grid.ga_functions.ga_unweighted`

**Signature**:
```python
predictions = get_unweighted_ensemble_predictions(
    best,
    ml_grid_object,
    valid=False
)
```

#### Function: `find_ensemble_weights_de`

Finds optimal weights for ensemble using Differential Evolution.

**Module**: `ml_grid.ga_functions.ga_ensemble_weight_finder_de`

**Signature**:
```python
weights = find_ensemble_weights_de(
    ensemble,
    ml_grid_object,
    valid=False
)
```







## Result Analysis APIs

### `ml_grid.util.GA_results_explorer`

Analyzes and visualizes experiment results.

#### Class: `GA_results_explorer`

Parses and visualizes GA experiment outcomes.

**Module**: `ml_grid.util.GA_results_explorer`

**Instantiation**:
```python
from ml_grid.util.GA_results_explorer import GA_results_explorer

explorer = GA_results_explorer(
    base_project_dir="HFE_GA_experiments",
)
```

**Methods**:

| Method | Parameters | Description |
|--------|------------|-------------|
| `plot_convergence()` | - | Generate fitness convergence plot |
| `plot_ensemble_size_performance()` | - | Performance vs. ensemble size |
| `plot_base_learner_importance()` | - | Feature/base learner importance |

---

### `ml_grid.util.evaluate_ensemble_methods`

Ensemble evaluation utilities (see [Model_API.md](Model_API.md) for base learner interface and [Pipeline_API.md](Pipeline_API.md) for GA pipeline context).

#### Class: `EnsembleEvaluator`

Evaluates ensembles on hold-out data.

**Module**: `ml_grid.util.evaluate_ensemble_methods`

**Instantiation**:
```python
from ml_grid.util.evaluate_ensemble_methods import EnsembleEvaluator

evaluator = EnsembleEvaluator(
    base_project_dir="HFE_GA_experiments",
    X_train=None, y_train=None,
    X_test=None, y_test=None,
    store_base_learners=True,
)
```

**Methods**:

| Method | Description |
|--------|-------------|
| `evaluate()` | Evaluate best ensemble on test set |



## Error Handling Reference

### Common Exceptions

| Exception | Cause | Resolution |
|-----------|-------|------------|
| `ModuleNotFoundError` | Package not installed | Run `pip install .` |
| `FileNotFoundError` | Dataset not found | Check `input_csv_path` in config |
| `ValueError: Input contains NaN` | Missing values | Adjust `percent_missing` threshold |

---

## Configuration YAML Schema

Complete list of configurable parameters:

```yaml
global_params:
  input_csv_path: str              # Required, path to dataset
  n_iter: int                      # Default: 20, grid search iterations
  model_list: List[str]            # Required, base learner names
  verbose: int                     # Default: 2, logging level (0-15)
  grid_n_jobs: int                 # Default: 8, parallel jobs
  base_project_dir: str            # Default: "HFE_GA_experiments"
  testing: bool                    # Default: False, use smaller grid
  test_sample_n: int               # Default: 0, no sampling

ga_params:
  nb_params: List[int]             # Default: [8, 16, 24], ensemble sizes
  pop_params: List[int]            # Default: [64, 128], population sizes
  g_params: List[int]              # Default: [100], generation counts

grid_params:
  weighted: List[str]              # Default: ["unweighted"], methods
  resample: List[str \| None]      # Default: [None], imbalancing handling
  corr: List[float]                # Default: [0.95], feature correlation
```

---

**Python Requirement**: Python >=3.12

This API documentation corresponds to version **v1.0+** of the ensemble_genetic_algorithm package.

For the latest API reference, please visit our [online documentation](https://ensemble-genetic-algorithm.readthedocs.io/).

 footnotes>
[1] API Documentation Summary: `API-Documentation-Summary.md`


