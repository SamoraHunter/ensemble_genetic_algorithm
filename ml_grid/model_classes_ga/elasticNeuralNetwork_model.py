import random
import time
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import ElasticNet
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.model_methods_ga import store_model


def elasticNeuralNetworkModelGenerator(ml_grid_object, local_param_dict):
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
    alpha_n = random.choice(
        [1, 0.5, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001, 0.0001]
    )
    max_iter_n = random.choice([5, 7, 10, 12, 15, 20, 25, 50, 75])
    l1_ratio_n = random.choice([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    loss_n = random.choice(["log"])
    penalty_n = random.choice(["elasticnet"])
    class_weight_n = random.choice(["balanced"])
    shuffle_n = random.choice([True])

    # warm_start = True
    model = OneVsRestClassifier(
        ElasticNet(
            alpha=alpha_n,  # untested OneVsRestClassifier
            max_iter=max_iter_n,
            l1_ratio=l1_ratio_n,
        )
    )

    # Fit model-------------------------------------------------------------------------------------
    model.fit(X_train, y_train)
    y_train_hat = model.predict(X_train)
    score = model.score(X_test, y_test)
    # print(score)
    y_pred = model.predict(X_test)
    mccscore = matthews_corrcoef(y_test, y_pred)

    auc_score = round(metrics.roc_auc_score(y_test, y_pred), 4)
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

    return (mccscore, model, list(X_train.columns), model_train_time, auc_score)
