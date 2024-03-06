from sklearn.metrics import make_scorer, roc_auc_score


class global_parameters():
    
    
    def __init__(self, debug_level=0, knn_n_jobs=-1):
        
        
        
        self.debug_level = debug_level
        
        self.knn_n_jobs = knn_n_jobs
        
        self.verbose = 3
        
        self.rename_cols = True
        
        self.error_raise = False
        
        self.random_grid_search = True

        self.sub_sample_param_space_pct = 0.001

        self.grid_n_jobs = 4

        self.metric_list = {'auc': make_scorer(roc_auc_score, needs_proba=False),
                        'f1':'f1',
                        'accuracy':'accuracy',
                        'recall': 'recall'}
        
        self.model_train_time_warning_threshold = 60
        
        self.store_base_learners = True
        
        self.gen_eval_score_threshold_early_stopping = 5
        
        self.log_store_dataframe_path = 'log_store_dataframe' 
    
        
    
    
        