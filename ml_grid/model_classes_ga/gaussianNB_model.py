import random
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.model_methods_ga import store_model

def GaussianNB_ModelGenerator(ml_grid_object, local_param_dict):
    global_parameter_val = global_parameters()
    
    verbose = global_parameter_val.verbose
    store_base_learners = ml_grid_object.global_params.store_base_learners
    
    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train
    y_test = ml_grid_object.y_test 
    
    start = time.time()
    
    X_train, X_test = feature_selection_methods_class(ml_grid_object).get_featured_selected_training_data(method='anova')
    
    # Initialise global parameter space----------------------------------------------------------------

    new_list = list(log_small).copy()
    new_list.append(1e-09)
    parameter_space = {
        "priors": [
            None,
            [0.1, 0.9],
            [0.9, 0.1],
            [0.7, 0.3],
            [0.3, 0.7],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.4, 0.6],
        ],  # enumerate
        "var_smoothing": new_list,
    }

    # Select a random sample from the global parameter space
    sample_parameter_space = {}
    for key in parameter_space.keys():
        sample_parameter_space[key] = random.choice(parameter_space.get(key))
        
    # fit model with random sample of global parameter space
    model = GaussianNB(**sample_parameter_space)

    # Train the model--------------------------------------------------------------------
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)
    mccscore = matthews_corrcoef(y_test, y_pred)
    
    auc_score = round(roc_auc_score(y_test,y_pred), 4)
    end = time.time()
    model_train_time = int(end-start)
    
    if(verbose >= 2):
        debug_base_learner(model, mccscore, X_train, auc_score, model_train_time)
    
    if(store_base_learners):
        store_model(ml_grid_object, local_param_dict, mccscore, model, list(X_train.columns), model_train_time, auc_score, y_pred)
    
    return (mccscore, model, list(X_train.columns), model_train_time, auc_score, y_pred)
