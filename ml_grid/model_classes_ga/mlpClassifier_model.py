import random
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.model_methods_ga import store_model
from ml_grid.util.param_space import ParamSpace


def MLPClassifier_ModelGenerator(ml_grid_object, local_param_dict):
    """Generates, trains, and evaluates an MLPClassifier model.

    This function performs a single trial of training and evaluating a
    Multi-layer Perceptron (MLP) classifier. It uses a random search
    approach for hyperparameter tuning.

    The process includes:
    1.  Applying ANOVA-based feature selection.
    2.  Randomly sampling hyperparameters from a comprehensive, predefined
        search space covering aspects like activation, alpha, learning rate,
        and network architecture.
    3.  Training the MLPClassifier with the selected parameters.
    4.  Evaluating the model's performance on the test set using Matthews
        Correlation Coefficient (MCC) and ROC AUC score.
    5.  Optionally storing the trained model and its metadata.

    Args:
        ml_grid_object: An object containing the project's data (e.g.,
            X_train, y_train, X_test, y_test) and configuration settings.
        local_param_dict (dict): A dictionary of local parameters for this
            specific model run, which may include 'param_space_size'.

    Returns:
        tuple: A tuple containing mccscore (float), the trained model object,
        a list of feature names, the model training time (int), the
        auc_score (float), and the predictions (np.ndarray).

    """
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

    param_dict = ParamSpace(size=local_param_dict.get("param_space_size")).param_dict

    log_small = param_dict["log_small"]
    bool_param = param_dict["bool_param"]
    log_large = param_dict["log_large"]
    log_large_long = param_dict["log_large_long"]
    log_med_long = param_dict["log_med_long"]
    log_med = param_dict["log_med"]
    log_zero_one = param_dict["log_zero_one"]
    lin_zero_one = param_dict["lin_zero_one"]

    # Initialise global parameter space----------------------------------------------------------------
    parameter_space = {
        "activation": ["relu"],
        "alpha": log_small,
        "batch_size": ["auto"],
        "beta_1": log_small,
        "beta_2": log_small,
        "early_stopping": bool_param,
        "epsilon": log_small,
        "hidden_layer_sizes": [
            5,
            10,
            50,
            100,
        ],  # log_large_long, # Possible DGX hang on 630
        "learning_rate": ["constant"],  # , "adaptive"],
        "learning_rate_init": log_small,
        "max_fun": [15000],
        "max_iter": log_large_long,
        "momentum": lin_zero_one,
        "n_iter_no_change": log_large_long,
        "nesterovs_momentum": [True],
        "power_t": [0.5],
        "random_state": [None],
        "shuffle": bool_param,
        "solver": ["adam", "lbfgs", "sgd"],
        "tol": log_small,
        "validation_fraction": [0.1],
        "verbose": [False],
        "warm_start": [False],
    }

    # Select a random sample from the global parameter space
    sample_parameter_space = {}
    for key in parameter_space.keys():
        sample_parameter_space[key] = random.choice(parameter_space.get(key))

    # fit model with random sample of global parameter space
    # display(sample_parameter_space)
    model = MLPClassifier(**sample_parameter_space)

    # Train the model--------------------------------------------------------------------
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)
    mccscore = matthews_corrcoef(y_test, y_pred)
    auc_score = round(roc_auc_score(y_test, y_pred), 4)
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

    return (mccscore, model, list(X_train.columns), model_train_time, auc_score, y_pred)
