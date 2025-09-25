"""Import parameter space constants"""

import numpy as np
from typing import Dict, List, Union

class ParamSpace:
    """
    Defines and provides access to predefined hyperparameter search spaces.

    This class holds different sized parameter dictionaries ('medium', 'xsmall', 'xwide')
    that can be used by model generators for hyperparameter tuning. Each dictionary
    contains various numpy arrays representing different scales and ranges of
    hyperparameters.
    """

    param_dict: Dict[str, Union[np.ndarray, List[bool]]]
    """A dictionary containing numpy arrays and lists for hyperparameter ranges."""

    def __init__(self, size: str):
        """
        Initializes the ParamSpace with a specific size.

        Args:
            size: The size of the parameter space to generate.
                Valid options are 'medium', 'xsmall', 'xwide'.
        """
        self.param_dict = None
        
        if(size == 'medium'):
            
            nstep = 3
            self.param_dict={    
                             
                'log_small': np.logspace(-1, -5, 3),
                'bool_param': [True, False],
                'log_large':np.logspace(0, 2, 3).astype(int),
                'log_large_long':np.floor(np.logspace(0, 3.1, 5)).astype(int),
                'log_med_long': np.floor(np.logspace(0, 1.5, 5)).astype(int),
                'log_med':np.floor(np.logspace(0, 1.5, 3)).astype(int),   
                'log_zero_one':np.logspace(0.0, 1.0, nstep) / 10,
                'lin_zero_one' : np.linspace(0.0, 1.0, nstep) / 10
                
            }
            
        
        if(size == 'xsmall'):
            
            nstep = 2
            self.param_dict={    
                             
                'log_small': np.logspace(-1, -5, 2),
                'bool_param': [True, False],
                'log_large':np.logspace(0, 2, 2).astype(int),
                'log_large_long':np.floor(np.logspace(0, 3.1, 2)).astype(int),
                'log_med_long': np.floor(np.logspace(0, 1.5, 2)).astype(int),
                'log_med':np.floor(np.logspace(0, 1.5, 2)).astype(int),   
                'log_zero_one':np.logspace(0.0, 1.0, nstep) / 10,
                'lin_zero_one' : np.linspace(0.0, 1.0, nstep) / 10
                
            }
            
        if(size == 'xwide'):
            
            nstep = 2
            self.param_dict={    
                             
                'log_small': np.logspace(-1, -5, 2),
                'bool_param': [True, False],
                'log_large':np.logspace(0, 2, 2).astype(int),
                'log_large_long':np.floor(np.logspace(0, 3.1, 2)).astype(int),
                'log_med_long': np.floor(np.logspace(0, 1.5, 2)).astype(int),
                'log_med':np.floor(np.logspace(0, 1.5, 2)).astype(int),   
                'log_zero_one':np.logspace(0.0, 1.0, nstep) / 10,
                'lin_zero_one' : np.linspace(0.0, 1.0, nstep) / 10
                
            }
            
            
            
            
  