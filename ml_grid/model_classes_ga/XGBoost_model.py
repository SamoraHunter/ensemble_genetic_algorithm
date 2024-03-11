import random
import time
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from xgboost import XGBClassifier
from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.model_methods_ga import store_model

def XGBoostModelGenerator(ml_grid_object, local_param_dict):
    global_parameter_val = global_parameters()
    
    verbose = global_parameter_val.verbose
    store_base_learners = ml_grid_object.global_params.store_base_learners
    
    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train
    y_test = ml_grid_object.y_test 
    
    start = time.time()
    
    X_train, X_test = feature_selection_methods_class(ml_grid_object).get_featured_selected_training_data(method='xgb')
    
    # Initialise parameter space-----------------------------------------------------------------
    gamma_n = random.choice([0.01, 0.1, 1, 3, 5, 7, 9, 10, 15])
    reg_alpha_n = random.choice([0, 0.001, 0.005, 0.01, 0.1, 1, 3, 5])
    reg_gamma_n = random.choice([0, 0.001, 0.005, 0.01, 0.1, 1, 3, 5])
    learning_rate_n = random.choice(
        [0.5, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    )
    subsample_n = random.choice([1, 0.98, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7])
    colsample_bytree_n = random.choice(
        [1, 0.98, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7])
    max_depth_n = random.choice([3, 4, 5, 6, 7, 8, 9, 10, 15])
    min_child_weight_n = random.choice(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    )
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
    try:
        gpu_id_n = get_free_gpu()
    except:
        gpu_id_n = '-1'
        pass
    
    try:
        model = XGBClassifier(
            gamma=gamma_n,
            reg_alpha=reg_alpha_n,
            learning_rate=learning_rate_n,
            subsample=subsample_n,
            colsample_bytree=colsample_bytree_n,
            max_depth=max_depth_n,
            min_child_weight=min_child_weight_n,
            n_estimators=n_estimators_n,
            tree_method="gpu_hist",
            gpu_id=gpu_id_n,
            verbosity=0,
            eval_metric='logloss',
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mccscore = matthews_corrcoef(y_test, y_pred)
        auc_score = round(roc_auc_score(y_test, y_pred), 4)
    except Exception as e:
        print("Error occurred:", e)
        model = XGBClassifier(
            gamma=gamma_n,
            reg_alpha=reg_alpha_n,
            learning_rate=learning_rate_n,
            subsample=subsample_n,
            colsample_bytree=colsample_bytree_n,
            max_depth=max_depth_n,
            min_child_weight=min_child_weight_n,
            n_estimators=n_estimators_n,
            tree_method="hist",
            verbosity=0,
            eval_metric='logloss',
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mccscore = matthews_corrcoef(y_test, y_pred)
        auc_score = round(roc_auc_score(y_test, y_pred), 4)
    
    end = time.time()
    model_train_time = int(end - start)
    
    if(verbose >= 2):
        debug_base_learner(model, mccscore, X_train, auc_score, model_train_time)
    
    if(store_base_learners):
        store_model(ml_grid_object, local_param_dict, mccscore, model, list(X_train.columns), model_train_time, auc_score, y_pred)
    
    return (mccscore, model, list(X_train.columns), model_train_time, auc_score)
