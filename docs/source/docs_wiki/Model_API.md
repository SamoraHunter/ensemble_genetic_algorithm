# Model API Reference

This document provides comprehensive API reference for all model classes in the **Ensemble Genetic Algorithm** project.

---

## Table of Contents

- [Base Learner Generator Interface](#base-learner-generator-interface)
- [Classification Models](#classification-models)
- [Model Generation Functions](#model-generation-functions)

See also: [GA_Python_API.md](GA_Python_API.md) for genetic algorithm evaluation methods and pipeline integration.

---

## Base Learner Generator Interface

### Abstract Interface

All base learner generators follow a consistent interface pattern:

**Required Methods**:
```python
class BaseLearnerGenerator:
    def hyperparameters(self, param_space):
        """Return dictionary of hyperparameter search spaces."""
        
    def model(self, params):
        """Instantiate model with given parameters."""
```

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

---

## Classification Models

### AdaBoost Classifier

#### Function: `AdaBoostClassifierModelGenerator`

Generates, trains, and evaluates an AdaBoost classifier.

**Module**: `ml_grid.model_classes_ga.adaboostClassifier_model`

**Generation Signature**:
```python
ada_boost_generator = AdaBoostClassifierModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ml_grid_object` | `Any` | Contains X_train, y_train, X_test, y_test and config |
| `local_param_dict` | `Dict` | Parameters for this specific run |

**Returns**: Same tuple format as base interface

---

### Decision Tree Classifier

#### Function: `DecisionTreeClassifierModelGenerator`

Generates, trains, and evaluates a decision tree classifier.

**Module**: `ml_grid.model_classes_ga.decisionTreeClassifier_model`

**Generation Signature**:
```python
dt_generator = DecisionTreeClassifierModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

---

### Dummy Model

#### Function: `DummyModelGenerator`

A baseline dummy model for comparison purposes.

**Module**: `ml_grid.model_classes_ga.dummy_model`

**Generation Signature**:
```python
dummy_generator = DummyModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

---

### Elastic Neural Network

#### Function: `elasticNeuralNetworkModelGenerator`

Generates an elastic neural network classifier.

**Module**: `ml_grid.model_classes_ga.elasticNeuralNetwork_model`

**Generation Signature**:
```python
enn_generator = elasticNeuralNetworkModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

---

### Extra Trees Classifier

#### Function: `extraTreesModelGenerator`

Generates, trains, and evaluates an Extra Trees classifier.

**Module**: `ml_grid.model_classes_ga.extra_trees_model`

**Generation Signature**:
```python
et_generator = extraTreesModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

---

### Gaussian NB

#### Function: `GaussianNB_ModelGenerator`

Generates, trains, and evaluates a Gaussian Naive Bayes classifier.

**Module**: `ml_grid.model_classes_ga.gaussianNB_model`

**Generation Signature**:
```python
gnb_generator = GaussianNB_ModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

---

### Gradient Boosting Classifier

#### Function: `GradientBoostingClassifier_ModelGenerator`

Generates, trains, and evaluates a gradient boosting classifier.

**Module**: `ml_grid.model_classes_ga.gradientBoostingClassifier_model`

**Generation Signature**:
```python
gb_generator = GradientBoostingClassifier_ModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

---

### K-Nearest Neighbors

#### Function: `kNearestNeighborsModelGenerator`

Generates, trains, and evaluates a k-nearest neighbors classifier.

**Module**: `ml_grid.model_classes_ga.kNearestNeighbors_model`

**Generation Signature**:
```python
knn_generator = kNearestNeighborsModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

---

### Logistic Regression

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

**Implementation Notes**:
- Uses ANOVA-based feature selection
- Random search for hyperparameters (C, max_iter, solver)
- Returns MCC and AUC scores

---

### MLP Classifier

#### Function: `MLPClassifier_ModelGenerator`

Generates, trains, and evaluates a multi-layer perceptron classifier.

**Module**: `ml_grid.model_classes_ga.mlpClassifier_model`

**Generation Signature**:
```python
mlp_generator = MLPClassifier_ModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

---

### Perceptron Model

#### Function: `perceptronModelGenerator`

Generates, trains, and evaluates a perceptron classifier.

**Module**: `ml_grid.model_classes_ga.perceptron_model`

**Generation Signature**:
```python
perc_generator = perceptronModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

---

### Perceptron Dummy Model

#### Function: `perceptron_dummy_model`

A baseline perceptron model for comparison purposes.

**Module**: `ml_grid.model_classes_ga.perceptron_dummy_model`

**Generation Signature**:
```python
perc_dummy_generator = perceptron_dummy_model(
    ml_grid_object,
    local_param_dict
)
```

---

### PyTorch ANN Binary Classifier

#### Function: `Pytorch_binary_class_ModelGenerator`

Generates, trains, and evaluates a PyTorch artificial neural network for binary classification.

**Module**: `ml_grid.model_classes_ga.pytorchANNBinaryClassifier_model`

**Generation Signature**:
```python
pytorch_generator = Pytorch_binary_class_ModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

---

### Quadratic Discriminant Analysis

#### Function: `QuadraticDiscriminantAnalysis_ModelGenerator`

Generates, trains, and evaluates a quadratic discriminant analysis classifier.

**Module**: `ml_grid.model_classes_ga.quadraticDiscriminantAnalysis_model`

**Generation Signature**:
```python
qda_generator = QuadraticDiscriminantAnalysis_ModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

---

### Random Forest

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

**Implementation Notes**:
- Uses RandomForest-based feature selection (see [Feature Selection](Data_Workflow.md#feature-selection))
- Random search for hyperparameters (n_estimators, max_features, max_depth)
- Returns MCC and AUC scores

See also: [Evaluation Methods](GA_Python_API.md#fitness-evaluation) for GA-based model evaluation

---

### SVC (Support Vector Classifier)

#### Function: `SVC_ModelGenerator`

Generates, trains, and evaluates a Support Vector Classifier.

**Module**: `ml_grid.model_classes_ga.svc_model`

**Generation Signature**:
```python
svc_generator = SVC_ModelGenerator(
    ml_grid_object,
    local_param_dict
)
```

---

### XGBoost Classifier

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

## Model Generation Functions

### Generic Pattern for All Models

All model generators follow this pattern:

```python
def model_nameModelGenerator(
    ml_grid_object: Any,
    local_param_dict: Dict
) -> Tuple[float, ModelClass, List[str], int, float, np.ndarray]:
    """Brief description of what this model does.
    
    The complete docstring includes:
    - Hyperparameter search strategy (random/grid)
    - Feature selection method used
    - Performance metrics calculated
    
    Args:
        ml_grid_object: Contains X_train, y_train, X_test, y_test and config
        local_param_dict: Parameters for this specific run
        
    Returns:
        Tuple of (mccscore, model, feature_names, train_time, auc_score, y_pred)
    """
    
    # Step 1: Apply feature selection
    X_train_selected, X_test_selected = apply_feature_selection(
        ml_grid_object, 
        method="anova"
    )
    
    # Step 2: Sample hyperparameters from space
    params = sample_hyperparameters()
    
    # Step 3: Initialize and train model
    model = create_model(params)
    model.fit(X_train_selected, y_train)
    
    # Step 4: Evaluate performance
    y_pred = model.predict(X_test)
    mccscore = matthews_corrcoef(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    
    # Step 5: Return results
    return (mccscore, model, feature_names, train_time, auc_score, y_pred)
```

---

## Model Registry

The `global_parameters` class contains a `MODEL_REGISTRY` that maps string names to model generator classes:

| Model Name | Generator Class |
|------------|-----------------|
| "AdaBoostClassifier" | `AdaBoostClassifierModelGenerator` |
| "DecisionTreeClassifier" | `DecisionTreeClassifierModelGenerator` |
| "elasticNeuralNetwork" | `elasticNeuralNetworkModelGenerator` |
| "extraTrees" | `extraTreesModelGenerator` |
| "GaussianNB" | `GaussianNB_ModelGenerator` |
| "GradientBoostingClassifier" | `GradientBoostingClassifier_ModelGenerator` |
| "kNearestNeighbors" | `kNearestNeighborsModelGenerator` |
| "logisticRegression" | `logisticRegressionModelGenerator` |
| "MLPClassifier" | `MLPClassifier_ModelGenerator` |
| "perceptron" | `perceptronModelGenerator` |
| "Pytorch_binary_class" | `Pytorch_binary_class_ModelGenerator` |
| "QuadraticDiscriminantAnalysis" | `QuadraticDiscriminantAnalysis_ModelGenerator` |
| "randomForest" | `randomForestModelGenerator` |
| "SVC" | `SVC_ModelGenerator` |
| "XGBoost" | `XGBoostModelGenerator` |
| "DummyModel" | `DummyModelGenerator` |

---

## Usage Example

```python
from ml_grid.util.global_params import global_parameters
from ml_grid.model_classes_ga.logistic_regression_model import (
    logisticRegressionModelGenerator
)

# Setup configuration
global_params = global_parameters(config_path='config.yml')

# Create model generator
model_gen = logisticRegressionModelGenerator(
    ml_grid_object=ml_grid_object,
    local_param_dict={'store_base_learners': True}
)

# Execute (returns tuple)
result = model_gen
mcc, model, features, train_time, auc, predictions = result

print(f"AUC: {auc:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"Training time: {train_time}s")
```

---

## Version Information

This API documentation corresponds to version **v1.0+** of the ensemble_genetic_algorithm package.

**Python Requirement**: Python >=3.12

To check your installed version:
```bash
pip show ensemble_genetic_algorithm
```
