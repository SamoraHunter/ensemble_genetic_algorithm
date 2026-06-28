# Genetic Algorithm Python API Reference

This document provides comprehensive API reference for the genetic algorithm Python interfaces in the **Ensemble Genetic Algorithm** project.

---

## Table of Contents

- [GA Core Functions](#ga-core-functions)
- [Fitness Evaluation](#fitness-evaluation)
- [Mutation Methods](#mutation-methods)
- [Weighting Methods](#weighting-methods)

---

## GA Core Functions

### `ml_grid.pipeline.ensemble_generator_ga.ensembleGenerator`

Generates an initial ensemble for the genetic algorithm.

#### Function: `ensembleGenerator`

Creates a new ensemble individual by selecting base learners from the model list.

**Module**: `ml_grid.pipeline.ensemble_generator_ga`

See also: [Configuration Guide](Configuration_Guide.md) for hyperparameter configuration options with `local_param_dict`.

**Signature**:
```python
individual = ensembleGenerator(
    nb_val,
    ml_grid_object
)
```

See also: [Pipeline API](Pipeline_API.md) for data pipeline setup with `ml_grid_object`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `nb_val` | `int` | Number of base learners in the ensemble |
| `ml_grid_object` | `Any` | Experiment object containing model list and configuration |

**Returns**: `list`
- Ensembles are represented as lists where `individual[0]` contains the base learners

---

### `ml_grid.pipeline.get_feature_selection_class_ga`

Feature selection utilities for GA pipeline.

#### Function: `feature_selection_methods_class.get_featured_selected_training_data`

Selects features based on various importance methods (see [Feature Selection](Data_Workflow.md#feature-selection) for details).

**Module**: `ml_grid.pipeline.get_feature_selection_class_ga`

See also: [Configuration Guide](Configuration_Guide.md) for `n_features` and `feature_selection_method` parameters, and [Data_Workflow.md](Data_Workflow.md#feature-selection) for feature selection details.

See also: [Pipeline API](Pipeline_API.md#feature-selection) for comprehensive feature selection utilities.

**Signature**:
```python
X_train_selected, X_test_selected = feature_selection_methods_class(
    ml_grid_object
).get_featured_selected_training_data(method="anova")
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"anova"` | Feature selection method: "anova", "randomforest", etc. |

**Returns**: `Tuple[pd.DataFrame, pd.DataFrame]`
- Selected features for training set
- Selected features for test set

---

## Fitness Evaluation

### `ml_grid.pipeline.evaluate_methods_ga.get_y_pred_resolver`

Resolves and generates predictions for ensemble evaluation (see [Pipeline_API](Pipeline_API.md) for GA pipeline overview).

#### Function: `get_y_pred_resolver`

Dispatches to the appropriate prediction method based on weighting configuration.

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
| `individual` | `List` | Required | Ensemble configuration (DEAP individual format) |
| `ml_grid_object` | `Any` | Required | Experiment object with data splits |
| `valid` | `bool` | `False` | If True, predict on validation set |

**Returns**: `Union[List, np.ndarray]`
- Final ensemble predictions

**Prediction Methods**:

The function dispatches based on `weighted` parameter in `local_param_dict`:

| Weighted Type | Method Called |
|---------------|---------------|
| `"unweighted"` | `get_unweighted_ensemble_predictions()` |
| `"de"` | `get_weighted_ensemble_prediction_de_y_pred_valid()` |
| `"ann"` | `get_y_pred_ann_torch_weighting()` |

---

### `ml_grid.pipeline.evaluate_methods_ga.evaluate_weighted_ensemble_auc`

Main fitness evaluation function for genetic algorithm (see [Pipeline_API](Pipeline_API.md) for GA pipeline context).

#### Function: `evaluate_weighted_ensemble_auc`

Evaluates an individual (ensemble) and returns fitness score.

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

**Evaluation Process**:

1. Generate predictions using `get_y_pred_resolver()`
2. Calculate performance metrics:
   - **AUC**: Receiver Operating Characteristic Area Under Curve
   - **MCC**: Matthews Correlation Coefficient
   - **F1**, **Precision**, **Recall**, **Accuracy**
3. Measure ensemble diversity using `measure_binary_vector_diversity()`
4. Apply diversity penalty if configured (`div_p > 0` - see [Configuration Guide](Configuration_Guide.md) for GA parameters)
5. Log results to CSV file

**Diversity Penalty**:

When `div_p > 0`, the fitness score is adjusted:
```python
auc_div, mcc_div = apply_diversity_penalty(
    auc, mcc, diversity_metric, diversity_params
)
```

**Diversity Parameters** (from config - see [Configuration Guide](Configuration_Guide.md)):
- `penalty_method`: `"linear"`, `"quadratic"`, `"exponential"`, or `"threshold"`
- `penalty_strength`: User-specified magnitude (default: 0.3)
- `min_score_factor`: Minimum score multiplier (default: 0.1)

---

### `ml_grid.pipeline.evaluate_methods_ga.normalize`

Normalizes weight vectors.

#### Function: `normalize`

L1-normalizes a vector of weights.

**Module**: `ml_grid.pipeline.evaluate_methods_ga`

**Signature**:
```python
normalized = normalize(weights)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `weights` | `np.ndarray` | Array of weights to normalize |

**Returns**: `np.ndarray`
- L1-normalized weight vector (sum of absolute values = 1)

---

### `ml_grid.pipeline.evaluate_methods_ga.measure_binary_vector_diversity`

Measures diversity between prediction vectors.

#### Function: `measure_binary_vector_diversity`

Calculates how diverse the predictions are across ensemble members.

**Module**: `ml_grid.pipeline.evaluate_methods_ga`

**Signature**:
```python
diversity = measure_binary_vector_diversity(
    ensemble,
    metric="jaccard"
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ensemble` | `List` | Required | Ensemble with base learners |
| `metric` | `str` | `"jaccard"` | Distance metric for diversity |

**Returns**: `float`
- Mean pairwise distance between prediction vectors

**Supported Metrics**:
- `"jaccard"`: Jaccard similarity distance
- `"hamming"`: Hamming distance
- Other scipy.spatial.distance metrics

---

## Mutation Methods

### `ml_grid.pipeline.mutate_methods.baseLearnerGenerator`

Generates a random base learner (see [Base Learner Interface](Model_API.md#base-learner-generator-interface) for generator interface details, and [Pipeline_API.md](Pipeline_API.md) for GA pipeline context).

#### Function: `baseLearnerGenerator`

Creates a new base learner by randomly selecting from the model list.

**Module**: `ml_grid.pipeline.mutate_methods`

**Signature**:
```python
new_learner = baseLearnerGenerator(ml_grid_object)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ml_grid_object` | `Any` | Experiment object with modelFuncList configuration |

**Returns**: `Any`
- Instance of a randomly selected model generator

---

### `ml_grid.pipeline.mutate_methods.mutateEnsemble`

Mutates an ensemble by replacing one base learner (see [Pipeline_API](Pipeline_API.md) for GA pipeline context).

#### Function: `mutateEnsemble`

Performs genetic mutation on an ensemble individual.

**Module**: `ml_grid.pipeline.mutate_methods`

**Signature**:
```python
mutated_individual = mutateEnsemble(
    individual,
    ml_grid_object
)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `individual` | `List` | Ensemble to mutate (individual[0] contains base learners) |
| `ml_grid_object` | `Any` | Experiment object with model configuration |

**Returns**: `List`
- Mutated individual with one base learner replaced

**Mutation Process**:

1. Randomly select a position in the ensemble
2. Remove the base learner at that position
3. Generate a new random baselearner using `baseLearnerGenerator()`
4. Append the new learner to complete mutation

---

## Weighting Methods

### Unweighted Ensemble Prediction

See also: [Pipeline_API.md](Pipeline_API.md) for GA pipeline workflow.

#### Function: `ml_grid.ga_functions.ga_unweighted.get_unweighted_ensemble_predictions`

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

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `best` | `List` | Required | Best ensemble configuration |
| `ml_grid_object` | `Any` | Required | Experiment object with data splits |
| `valid` | `bool` | `False` | Use validation set if True |

**Returns**: `List`
- Final predictions via mode of individual model predictions

---

### DE Weighted Ensemble Prediction

See also: [Pipeline_API.md](Pipeline_API.md) for pipeline workflow and [Configuration_Guide.md](Configuration_Guide.md) for weighted prediction options (`"de"` method).

#### Function: `ml_grid.ga_functions.ga_de_weight_method.get_weighted_ensemble_prediction_de_y_pred_valid`

Generates predictions using Differential Evolution weighted ensemble.

**Module**: `ml_grid.ga_functions.ga_de_weight_method`

**Signature**:
```python
predictions = get_weighted_ensemble_prediction_de_y_pred_valid(
    ensemble,
    weights,
    ml_grid_object,
    valid=False
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ensemble` | `List` | Required | Ensemble configuration |
| `weights` | `np.ndarray` | Required | Learner weights from optimization |
| `ml_grid_object` | `Any` | Required | Experiment object |
| `valid` | `bool` | `False` | Use validation set if True |

**Returns**: `Union[List, np.ndarray]`
- Weighted ensemble predictions

---

### DE Weight Ensemble Finder

See also: [Pipeline_API.md](Pipeline_API.md) for pipeline workflow and [Configuration_Guide.md](Configuration_Guide.md) for weight optimization parameters.

#### Function: `ml_grid.ga_functions.ga_ensemble_weight_finder_de.find_ensemble_weights_de`

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

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ensemble` | `List` | Required | Ensemble to optimize weights for |
| `ml_grid_object` | `Any` | Required | Experiment object |
| `valid` | `bool` | `False` | Optimize on validation set if True |

**Returns**: `np.ndarray`
-Optimized weight vector

---

### ANN Weighted Ensemble Prediction

See also: [Pipeline_API.md](Pipeline_API.md) for pipeline workflow and [Configuration_Guide.md](Configuration_Guide.md) for weighted prediction options (`"ann"` method).

#### Function: `ml_grid.ga_functions.ga_ann_weight_methods.get_y_pred_ann_torch_weighting`

Generates predictions using neural network-based ensemble weighting.

**Module**: `ml_grid.ga_functions.ga_ann_weight_methods`

**Signature**:
```python
predictions = get_y_pred_ann_torch_weighting(
    ensemble,
    ml_grid_object,
    valid=False
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ensemble` | `List` | Required | Ensemble configuration |
| `ml_grid_object` | `Any` | Required | Experiment object |
| `valid` | `bool` | `False` | Use validation set if True |

**Returns**: `Union[List, np.ndarray]`
- Neural network-weighted predictions

---

## Utility Functions

### `ml_grid.util.ensemble_diversity_methods`

Diversity measurement and penalty application.

#### Function: `measure_diversity_wrapper`

Wrapper for ensemble diversity measurement.

**Module**: `ml_grid.util.ensemble_diversity_methods`

**Signature**:
```python
diversity = measure_diversity_wrapper(
    individual,
    method="comprehensive"
)
```

#### Function: `apply_diversity_penalty`

Applies diversity-based penalty to fitness scores.

**Module**: `ml_grid.util.ensemble_diversity_methods`

**Signature**:
```python
auc_penalized, mcc_penalized = apply_diversity_penalty(
    auc_score,
    mcc_score,
    diversity_metric,
    params=diversity_params
)
```

---

## Version Information

This API documentation corresponds to version **v1.0+** of the ensemble_genetic_algorithm package.

**Python Requirement**: Python >=3.12

To check your installed version:
```bash
pip show ensemble_genetic_algorithm
```
