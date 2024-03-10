import random
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.model_methods_ga import store_model


def perceptronModelGenerator(ml_grid_object, local_param_dict):
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
    y_train_hat = model.predict(X_train)  # predict
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

    return (
        (mccscore, model, list(X_train.columns), model_train_time, auc_score, y_pred),
    )
