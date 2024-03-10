import random
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.model_methods_ga import store_model


def randomForestModelGenerator(ml_grid_object, local_param_dict):
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
    ).get_featured_selected_training_data(method="randomforest")

    # Initialise parameter space-----------------------------------------------------------------
    num_trees = 30
    max_features_n = random.choice(list(np.arange(0.05, 0.2, 0.001)))
    min_samples_leaf_n = random.choice([2, 5, 8, 10, 15, 20, 25, 40, 50])
    n_estimators_n = random.choice(
        [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            125,
            150,
            175,
            200,
            225,
            250,
            275,
            300,
            325,
            350,
            375,
            400,
            425,
            450,
            475,
            500,
            550,
            600,
            650,
            700,
        ]
    )
    max_depth_n = random.choice([2, 4, 8, 10, None])
    class_weight_n = random.choice(["balanced"])

    # warm_start = True
    model = RandomForestClassifier(
        n_estimators=n_estimators_n,
        max_features=max_features_n,
        min_samples_leaf=min_samples_leaf_n,
        max_depth=max_depth_n,
        class_weight=class_weight_n,
    )

    # Fit model-------------------------------------------------------------------------------------
    model.fit(X_train, y_train)
    y_train_hat = model.predict(X_train)
    score = model.score(X_test, y_test)
    # print(score)
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
