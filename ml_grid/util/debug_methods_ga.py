

from ml_grid.util.global_params import global_parameters


def debug_base_learner(model, mccscore, X_train, auc_score, model_train_time):
    
    global_parameters_vals = global_parameters()
    
    verbose = global_parameters_vals.verbose
    
    model_train_time_warning_threshold = global_parameters_vals.model_train_time_warning_threshold
    
    print(str(model).split("(")[0], round(mccscore, 5), len(X_train.columns), auc_score, model_train_time)
    if(model_train_time> model_train_time_warning_threshold):
        print("Warning long train time, ")
        
        
        