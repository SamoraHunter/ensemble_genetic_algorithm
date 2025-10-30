# Adding a New Base Learner

This guide explains how to extend the **Ensemble Genetic Algorithm** project by adding your own custom machine learning model as a base learner. This allows the genetic algorithm to include your model when building and evolving ensembles.

---

## Overview

The framework is designed to be extensible. Any model that follows a scikit-learn compatible API (`.fit()`, `.predict_proba()`) can be integrated. To do this, you need to create a "Model Generator" class. This class acts as a wrapper that tells the framework how to:

1.  Define the hyperparameter search space for your model.
2.  Instantiate your model with a given set of hyperparameters.

## The Model Generator Class Structure

A model generator class must have the following structure:

-   An `__init__` method that accepts `ml_grid_object` and `local_param_dict`.
-   A `get_hyperparameter_space` method that returns the hyperparameter search space for the model.
-   A `get_model` method that returns an instance of your model, configured with specific hyperparameters.

## Step-by-Step Guide

Let's walk through adding a `SGDClassifier` from scikit-learn as a new base learner.

### Step 1: Create the Model Generator File

Create a new Python file in the `ml_grid/model_classes_ga/` directory. Let's call it `sgd_classifier_model.py`.

### Step 2: Define the Class and its Methods

In your new file, define the `SGDClassifierModelGenerator` class with the required methods. The complete code for the file should look like this:

```python
from sklearn.linear_model import SGDClassifier
from hyperopt import hp
import numpy as np

class SGDClassifierModelGenerator:
    """
    A model generator for the scikit-learn SGDClassifier.
    """
    def __init__(self, ml_grid_object, local_param_dict):
        self.ml_grid_object = ml_grid_object
        self.local_param_dict = local_param_dict
        # Access global parameters like random_state
        self.global_param_dict = ml_grid_object.global_param_dict

    def get_hyperparameter_space(self):
        """
        Returns the hyperparameter search space for SGDClassifier.
        """
        return {
            'loss': hp.choice('loss', ['hinge', 'log_loss', 'modified_huber']),
            'penalty': hp.choice('penalty', ['l2', 'l1', 'elasticnet']),
            'alpha': hp.loguniform('alpha', np.log(0.0001), np.log(0.1)),
            'max_iter': hp.choice('max_iter', [1000, 2000, 3000]),
            'tol': hp.loguniform('tol', np.log(1e-4), np.log(1e-2)),
        }

    def get_model(self, param_dict):
        """
        Returns an initialized SGDClassifier model instance.
        """
        model = SGDClassifier(
            loss=param_dict['loss'],
            penalty=param_dict['penalty'],
            alpha=param_dict['alpha'],
            max_iter=param_dict['max_iter'],
            tol=param_dict['tol'],
            random_state=self.global_param_dict.get('random_state'),
            class_weight='balanced' # Often a good default for classification
        )
        return model
```
**Note**: The `log_loss` or `modified_huber` options for the `loss` parameter are important, as they enable `SGDClassifier` to provide probability estimates via `predict_proba()`, which is required by the ensemble methods.

### Step 3: Integrate into the Experiment

Now, you can use your new model generator in your experiments. The framework automatically discovers model generators based on the names provided in your `config.yml`.

1.  **Update `config.yml`**: Add the name of your new model to the `model_list` under `global_params`. The name should match the class name of your model generator.

    ```yaml
    # in config.yml
    global_params:
      model_list:
        - "LogisticRegressionModelGenerator"
        - "RandomForestModelGenerator"
        - "XGBoostModelGenerator"
        - "SGDClassifierModelGenerator" # Add your new model's class name
    ```

That's it! When you run an experiment using `main.py`, the framework will now be able to select, tune, and include `SGDClassifier` in the ensembles it evolves. You can follow this same pattern to add any scikit-learn compatible classifier to the project.

---

## Creating Your Own Custom Classifiers

Beyond using existing scikit-learn models, you can also create your own custom estimators from scratch. As long as your custom class adheres to the scikit-learn API (implementing `.fit()`, `.predict_proba()`, `get_params()`, and `set_params()`), it can be integrated into this framework using the same model generator pattern.

This allows you to experiment with novel algorithms while still leveraging the power of the genetic algorithm for ensembling and hyperparameter tuning.

For detailed instructions on how to develop your own scikit-learn compatible estimators, refer to the official scikit-learn developer's guide.