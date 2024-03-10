import random
import time

from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.model_methods_ga import store_model
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def logisticRegressionModelGenerator(ml_grid_object, local_param_dict):

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

    # Initialise parameter space-----------------------------------------------------------------
    solver_n = random.choice(["sag"])
    class_weight_n = random.choice(["balanced"])
    max_iter_n = random.choice([5, 7, 10, 12, 15, 20, 25, 50, 75])
    C_n = random.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e1, 1, 1e1, 1e2, 1e3, 1e4, 1e5])
    model = LogisticRegression(
        solver=solver_n, class_weight=class_weight_n, max_iter=max_iter_n, C=C_n
    )

    # Fit model---------------------------------------------------------------------
    model.fit(X_train, y_train)
    y_train_hat = model.predict(X_train)
    score = model.score(X_test, y_test)
    # print(score)
    y_pred = model.predict(X_test)
    mccscore = metrics.matthews_corrcoef(y_test, y_pred)
    auc_score = round(metrics.roc_auc_score(y_test, y_pred), 4)
    end = time.time()
    model_train_time = int(end - start)
    if verbose >= 2:
        debug_base_learner(model, mccscore, X_train, auc_score, model_train_time)
        # print(sample_parameter_space)

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
