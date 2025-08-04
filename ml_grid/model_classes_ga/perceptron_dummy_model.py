import time
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.global_params import global_parameters


def perceptronModelGen_dummy(ml_grid_object, local_param_dict):
    """Generates and trains a simple Perceptron model for dummy/baseline purposes.

    This function creates a basic Perceptron classifier with a fixed set of
    hyperparameters. It is primarily intended to be used by the
    `DummyModelGenerator` to provide a consistent, pre-trained model object
    that can be used as a placeholder in ensembles.

    The process includes:
    1.  Applying ANOVA-based feature selection.
    2.  Scaling the feature data using StandardScaler.
    3.  Training a Perceptron model with a randomly chosen `max_iter` from a
        small, predefined list.
    4.  Evaluating the model's performance on the test set.

    Args:
        ml_grid_object: An object containing the project's data (e.g.,
            X_train, y_train, X_test, y_test) and configuration settings.
        local_param_dict (dict): A dictionary of local parameters, maintained
            for API consistency but not directly used for hyperparameter tuning.

    Returns:
        tuple: A tuple containing mccscore (float), the trained model object,
        a list of feature names, the model training time (int), the
        auc_score (float), and the predictions (np.ndarray).

    """

    global_parameter_val = global_parameters()

    verbose = global_parameter_val.verbose
    store_base_learners = ml_grid_object.global_params.store_base_learners
    # scale = ml_grid_object.scale

    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train
    y_test = ml_grid_object.y_test

    debug = ml_grid_object.verbose

    start = time.time()

    X_train, X_test = feature_selection_methods_class(
        ml_grid_object
    ).get_featured_selected_training_data(method="anova")
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Dummy model initialization
    maxIterList = [5, 7, 10, 12, 15, 20, 25, 50, 75]
    model = Perceptron(max_iter=random.choice(maxIterList), eta0=0.1, random_state=0)

    # Train the perceptron
    model.fit(X_train_std, y_train)

    # Predict
    y_pred = model.predict(X_test_std)
    mccscore = matthews_corrcoef(y_test, y_pred)
    try:
        auc_score = round(roc_auc_score(y_test, y_pred), 4)
    except ValueError:
        auc_score = 0.5

    end = time.time()
    model_train_time = int(end - start)

    if debug:
        debug_base_learner(model, mccscore, X_train, auc_score, model_train_time)

    return (mccscore, model, list(X_train.columns), model_train_time, auc_score, y_pred)
