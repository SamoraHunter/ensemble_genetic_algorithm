import random
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn import metrics
from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.model_methods_ga import store_model


def perceptronModelGenerator(
    ml_grid_object: Any, local_param_dict: Dict
) -> Tuple[float, Perceptron, List[str], int, float, np.ndarray]:
    """Generates, trains, and evaluates a Perceptron model.

    This function performs a single trial of training and evaluating a
    Perceptron model. It uses a simple random search for the `max_iter`
    hyperparameter.

    The process includes:
    1.  Applying ANOVA-based feature selection.
    2.  Scaling the feature data using StandardScaler.
    3.  Randomly sampling the `max_iter` hyperparameter from a predefined list.
    4.  Training the Perceptron model with the selected parameters.
    5.  Evaluating the model's performance on the test set using Matthews
        Correlation Coefficient (MCC) and ROC AUC score.
    6.  Optionally storing the trained model and its metadata.

    Args:
        ml_grid_object (Any): An object containing the project's data (e.g.,
            X_train, y_train, X_test, y_test) and configuration settings.
        local_param_dict (Dict): A dictionary of local parameters for this
            specific model run (used for model storage).

    Returns:
        A tuple containing the following elements:
            - mccscore (float): The Matthews Correlation Coefficient.
            - model (Perceptron): The trained model object.
            - feature_names (List[str]): A list of feature names used for training.
            - model_train_time (int): The model training time in seconds.
            - auc_score (float): The ROC AUC score.
            - y_pred (np.ndarray): The model's predictions on the test set.

    """
    from ml_grid.util.global_params import global_parameters
    global_parameter_val = global_parameters()


    verbose = global_parameter_val.verbose
    store_base_learners = ml_grid_object.global_params.store_base_learners

    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train
    y_test = ml_grid_object.y_test

    start = time.time()

    X_train, X_test = feature_selection_methods_class(
        ml_grid_object
    ).get_featured_selected_training_data(method="anova")

    # Initialise parameter space----------------------------------------------------------------
    maxIterList = [5, 7, 10, 12, 15, 20, 25, 50, 75]
    sc = StandardScaler()
    sc.fit(X_train)
    # Apply the scaler to the X training data
    X_train_std = sc.transform(X_train)

    # Apply the SAME scaler to the X test data
    X_test_std = sc.transform(X_test)

    # Create a perceptron object with the parameters: n iterations (epochs) over the data, and a learning rate of 0.1
    model = Perceptron(max_iter=random.choice(maxIterList), eta0=0.1, random_state=0)

    # Train the perceptron--------------------------------------------------------------------
    model.fit(X_train_std, y_train)
    y_train_hat = model.predict(X_train_std)  # predict
    y_pred = model.predict(X_test_std)
    mccscore = matthews_corrcoef(y_test, y_pred)
    try:
        auc_score = round(metrics.roc_auc_score(y_test, y_pred), 4)
    except ValueError:
        auc_score = 0.5

    end = time.time()
    model_train_time = int(end - start)

    if verbose >= 2:
        debug_base_learner(model, mccscore, X_train, auc_score, model_train_time)

    if store_base_learners:
        store_model(
            ml_grid_object,
            local_param_dict,
            mccscore,
            model,
            list(X_train.columns),
            model_train_time,
            auc_score,
            y_pred,
        )

    if ml_grid_object.verbose >= 1:
        print("")
        print(
            mccscore, model, list(X_train.columns), model_train_time, auc_score, y_pred
        )

    return (
        mccscore,
        model,
        list(X_train.columns),
        model_train_time,
        auc_score,
        y_pred,
    )
