import random
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.model_methods_ga import store_model

def kNearestNeighborsModelGenerator(ml_grid_object, local_param_dict):
    """Generates, trains, and evaluates a K-Nearest Neighbors (KNN) classifier.

    This function performs a single trial of training and evaluating a
    KNeighborsClassifier. It uses a random search approach for
    hyperparameter tuning.

    The process includes:
    1.  Applying ANOVA-based feature selection.
    2.  Randomly sampling hyperparameters ('n_neighbors', 'weights') from a
        predefined search space.
    3.  Validating that 'n_neighbors' does not exceed the number of training
        samples.
    4.  Training the KNeighborsClassifier with the selected parameters.
    5.  Evaluating the model's performance on the test set using Matthews
        Correlation Coefficient (MCC) and ROC AUC score.
    6.  Optionally storing the trained model and its metadata.

    Args:
        ml_grid_object: An object containing the project's data (e.g.,
            X_train, y_train, X_test, y_test) and configuration settings.
        local_param_dict (dict): A dictionary of local parameters for this
            specific model run (unused in this function but maintained for
            API consistency).

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
    
    X_train, X_test = feature_selection_methods_class(ml_grid_object).get_featured_selected_training_data(method='anova')
    
    # Initialise parameter space-----------------------------------------------------------------
    n_neighbours_n = random.choice(
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100]
    )
    
    mccscore = 0
    auc_score = 0.5
    model_train_time = 0

    if(n_neighbours_n > len(X_train)):
        n_neighbours_n = len(X_train)-1
        print("warning kNearestNeighborsModelGen", "nn > sample")
    
    weights_n = random.choice(["uniform", "distance"])
    try:
        model = KNeighborsClassifier(n_neighbors=n_neighbours_n, weights=weights_n, n_jobs=1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mccscore = matthews_corrcoef(y_test, y_pred)
        auc_score = round(roc_auc_score(y_test,y_pred), 4)
    except Exception as e:
        print("Error occurred:", e)
    
    end = time.time()
    model_train_time = int(end-start)
    
    if(verbose >= 2):
        debug_base_learner(model, mccscore, X_train, auc_score, model_train_time)
    
    if(store_base_learners):
        store_model(ml_grid_object, local_param_dict, mccscore, model, list(X_train.columns), model_train_time, auc_score, y_pred)
    
    return (mccscore, model, list(X_train.columns), model_train_time, auc_score, y_pred)
