

import json
import pickle
import random
import time

import numpy as np
import torch


def store_model(ml_grid_object, local_param_dict, mccscore, model, feature_list, model_train_time, auc_score, y_pred, model_type='sklearn'): #**kwargs ):

    
    
    model_store_path = ml_grid_object.logging_paths_obj.model_store_path
    
    base_project_dir = ml_grid_object.logging_paths_obj.base_project_dir
    
    global_param_str = ml_grid_object.logging_paths_obj.global_param_str
    
    additional_naming = ml_grid_object.logging_paths_obj.additional_naming
    
    global_param_dict = ml_grid_object.global_params
    
    log_folder_path = ml_grid_object.logging_paths_obj.log_folder_path
    
    
    with open(model_store_path, 'r') as f:
        model_store_data = json.load(f)
    
    idx = len(model_store_data['models']) + 1
    
    time_stamp =  time.time_ns() 
    
    
    if(model_type=='sklearn'):
        model = str(model)
    
    elif(model_type=='torch'):
        y_pred = y_pred.astype(float)
        torch.save(model, f=f"{log_folder_path}/"+"/torch/"+str(time_stamp))
        model=time_stamp
        
    elif(model_type=='xgb'):    
        pickle.dump(model, open(f"{log_folder_path}/"+"/xgb/"+str(time_stamp), "wb"))
        model=time_stamp
        y_pred = y_pred.astype(float)
        
    #print(type(model))
    scale = local_param_dict.get('scale')
    if(scale):
        y_pred = y_pred.astype(float)
    
    model_store_entry = {
        'index':idx,
        'mcc_score':mccscore,
        'model': model, 
        'feature_list':feature_list,
        'model_train_time':model_train_time,
        'auc_score':auc_score,
        'y_pred': list(y_pred),
        'model_type': model_type,

    }
    #print(model_store_entry)
    
    model_store_data['models'].update({idx: model_store_entry})
    
    jsonString = json.dumps(model_store_data)
    jsonFile = open(model_store_path, "w", encoding='utf-8')
    jsonFile.write(jsonString)
    jsonFile.close()
    
    torch.cuda.empty_cache() #exp
    
    
    
    
def get_stored_model(ml_grid_object):
    
    
    model_store_path = ml_grid_object.logging_paths_obj.model_store_path
    
    global_param_str = ml_grid_object.logging_paths_obj.global_param_str
    
    additional_naming = ml_grid_object.logging_paths_obj.additional_naming
    
    global_param_dict = ml_grid_object.global_params
    
    log_folder_path = ml_grid_object.logging_paths_obj.log_folder_path
    
    
    with open(model_store_path, 'r', encoding='utf-8') as f:
        model_store_data = json.load(f)
    
    model_key_list = list(model_store_data['models'].keys())
    
    try:
        model_key = str(random.choice(model_key_list))

        print(f"Returning stored model at index {model_key}/{len(model_key_list)}")
        
         
        
        if(model_store_data['models'].get(model_key)['model_type'] == 'sklearn'):
            model = eval(model_store_data['models'].get(model_key)['model'])
            
        elif(model_store_data['models'].get(model_key)['model_type'] == 'torch'):
            time_stamp = model_store_data['models'].get(model_key)['model']
            model = torch.load(f=f"{log_folder_path}/"+"/torch/"+str(time_stamp))
            
        elif(model_store_data['models'].get(model_key)['model_type'] == 'xgb'):
            time_stamp = model_store_data['models'].get(model_key)['model']
            model = pickle.load(open(f"{log_folder_path}/"+"/xgb/"+str(time_stamp), "rb"))
            
        

        return (model_store_data['models'].get(model_key)['mcc_score'],
                model,
               model_store_data['models'].get(model_key)['feature_list'],
               model_store_data['models'].get(model_key)['model_train_time'],
               model_store_data['models'].get(model_key)['auc_score'],
                np.array(model_store_data['models'].get(model_key)['y_pred'])
               )
    except Exception as e:
        print("Failed inside getting stored model, returning random new model")
        index = random.randint(0, len(modelFuncList)-1)


        return modelFuncList[index]()
    
    
    