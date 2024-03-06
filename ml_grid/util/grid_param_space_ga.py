
import itertools as it
import random

from ml_grid.util.global_params import global_parameters


class Grid:



    def __init__(self, sample_n=1000, test_grid=False):
        
        self.test_grid = test_grid
        
        self.global_params = global_parameters()
        
        self.verbose = self.global_params.verbose

        if(sample_n == None):
            self.sample_n = 1000
        else:
            self.sample_n = sample_n
            
        if(self.verbose >=1):    
            print(f"Feature space slice sample_n {self.sample_n}")
        #Default grid
        #User can update grid dictionary on the object
        self.grid = {
    'weighted': ['ann', 'de', 'unweighted'],
    'use_stored_base_learners':[False],
    'store_base_learners':[False], 
    'resample' : ['undersample'],
    'scale'    : [True],
    'n_features': ['all'],
    'param_space_size':['medium'],
    'n_unique_out': [10],
    'outcome_var_n':['1'],
    'div_p':[0],
    'percent_missing':[99.9, 95, 90],  #n/100 ex 95 for 95%
                     'corr':[0.9, 0.99],
                      'cxpb':[0.5, 0.75, 0.25],
                       'mutpb':[0.2, 0.4, 0.8],
                        'indpb':[0.025, 0.05, 0.075],
                        't_size':[3, 6, 9],
                         'data':[
                             {'age':[True],
                            'sex':[True],
                            'bmi':[True],
                            'ethnicity':[True],
                            'bloods':[True],
                            'diagnostic_order':[True],
                            'drug_order':[True],
                            'annotation_n':[True],
                            'meta_sp_annotation_n':[True],
                            'annotation_mrc_n':[True, False],
                            'meta_sp_annotation_mrc_n':[True],
                            'core_02':[True],
                            'bed':[True],
                            'vte_status':[True],
                            'hosp_site':[True],
                            'core_resus':[True],
                            'news':[True],
                            'date_time_stamp':[True]}
                         ]
}
        
        
        def c_prod(d):
            if isinstance(d, list):
                for i in d:
                    yield from ([i] if not isinstance(i, (dict, list)) else c_prod(i))
            else:
                for i in it.product(*map(c_prod, d.values())):
                    yield dict(zip(d.keys(), i))


        self.settings_list = list(c_prod(self.grid))
        print(f"Full settings_list size: {len(self.settings_list)}")
        
        
        random.shuffle(self.settings_list)
        
        self.settings_list = random.sample(self.settings_list, self.sample_n)
        
        self.settings_list_iterator = iter(self.settings_list)
        
        #This is likely not properly functioning. Does not return iteration, instead reinitiates. 
        #Don't need to subsample, can just generate n number of random choices from grid space. 
        #function can just return random choice from grid space, terminate at the other end once limit reached. 
        
        
        
        
        
        #test space

        # nb_params = [4, 8, 16]
        # pop_params = [10, 20]
        # g_params = [10, 30]


        # nb_params = [4, 8, 16, 32]
        # pop_params = [32, 64, 128]
        # g_params = [128]

        self.nb_params = [16]
        self.pop_params = [32]
        self.g_params = [128]
        
        if(self.test_grid):
            self.nb_params = [4, 8, 16]
            self.pop_params = [10, 20]
            self.g_params = [10, 30]