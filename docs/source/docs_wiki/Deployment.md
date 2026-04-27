# Model Deployment Guide

This guide explains how to take the best ensemble model discovered by the genetic algorithm and deploy it for production use as a portable, scikit-learn compatible object.

---

## Overview

The genetic algorithm produces ensembles composed of various base learners, each with its own hyperparameters and feature subset. To make this "portable," the framework provides a scikit-learn compatible wrapper called `SklearnEnsembleClassifier`. 

This wrapper encapsulates the logic for feature masking (ensuring each base model only sees its specific features) and prediction averaging, allowing you to treat the entire ensemble as a single estimator.

## The Deployment Workflow

Deploying a model involves four main steps:

1.  **Identify the Best Run**: Locate your experiment results in `final_grid_score_log.csv` and pick the iteration with the highest performance metric (usually `auc`).
2.  **Reconstruct the Ensemble**: Use the `EnsembleEvaluator` to "re-hydrate" the architecture string back into functional Python objects.
3.  **Final Fit**: Train the constituent base learners on your training dataset.
4.  **Serialize**: Save the fitted ensemble to a file using `joblib`.

### Example Reconstruction

```python
from ml_grid.util.evaluate_ensemble_methods import EnsembleEvaluator
from ml_grid.util.ensemble_classifier import SklearnEnsembleClassifier
import joblib
import pandas as pd

# 1. Load results and find best ensemble
results_df = pd.read_csv("path/to/final_grid_score_log.csv")
best_row = results_df.loc[results_df['auc'].idxmax()]

# 2. Initialize evaluator to parse the architecture string
evaluator = EnsembleEvaluator(...)
parsed_arch = evaluator._parse_ensemble(best_row['best_ensemble'])[0]

# 3. Wrap in the Sklearn classifier and fit
my_ensemble = SklearnEnsembleClassifier(parsed_arch, evaluator.original_feature_names)
my_ensemble.fit(evaluator.ml_grid_object.X_train, evaluator.ml_grid_object.y_train)

# 4. Save the model
joblib.dump(my_ensemble, "deployed_ensemble_model.joblib")
```

## Production Environment

To run the model on another server, the target environment requires the following:

-   **Python 3.10+**
-   **Core Libraries**: `numpy`, `pandas`, `scikit-learn`, `joblib`.
-   **PyTorch**: Required if your ensemble includes neural network base learners (`BinaryClassification`).
-   **Project Package**: The `ensemble-genetic-algorithm` package must be installed so the environment can resolve the custom class definitions during deserialization.

```bash
# Install from the local repository as the package is not on PyPI yet
./setup.sh --cpu
```

## Running Predictions in Production

Once the `.joblib` file is transferred and the environment is ready, you can use the model with minimal code:

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('deployed_ensemble_model.joblib')

# Prepare data (must be a DataFrame with original feature names)
new_data = pd.read_csv('production_data.csv')

# Get predictions and probabilities
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)
```

**Note**: The `SklearnEnsembleClassifier` is robust to column ordering; as long as the required feature names are present in the input DataFrame, it will correctly subset the data for each base learner internally.