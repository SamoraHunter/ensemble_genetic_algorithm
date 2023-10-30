

import json
import logging
import os
import pathlib


class log_folder():


        def __init__(self, local_param_dict, additional_naming, base_project_dir):
            
            
            self.additional_naming = additional_naming
            self.base_project_dir = base_project_dir
            
            
            str_b = ''
            for key in local_param_dict.keys():
                if(key != 'data'):
                    str_b = str_b + '_' + str(local_param_dict.get(key))
                else:
                    for key in local_param_dict.get('data'):
                        str_b = str_b + str(int(local_param_dict.get('data').get(key)))
            
            self.global_param_str = str_b
            #self.global_param_str = str(global_param_dict).replace("{", "").replace("}", "").replace(":", "").replace(" ", "").replace(",", "").replace("'", "_").replace("__", "_").replace("'","").replace(",","").replace(": ", "_").replace("{","").replace("}","").replace("True","T").replace("False", "F").replace(" ","_").replace("[", "").replace("]", "").replace("_","")
            
            print(self.global_param_str)

            
            self.log_folder_path = f"{self.global_param_str + additional_naming}/logs/"
            
            pathlib.Path(self.base_project_dir+self.log_folder_path).mkdir(parents=True, exist_ok=True) 
            
            #full_log_path = f"{self.base_project_dir+self.global_param_str + additional_naming}/logs/log.log"
            
            full_log_path = f"{self.base_project_dir+self.global_param_str + additional_naming}"
            
            
            self.log_folder_path = full_log_path
            
            try:
                logging.basicConfig(filename=full_log_path)
                stderrLogger=logging.StreamHandler()
                stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
                logging.getLogger().addHandler(stderrLogger)
            except Exception as e:
                print("Failed to set log dir at ", full_log_path)
                print(e)
            
            

            try:
                pathlib.Path(self.log_folder_path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"An error occurred while creating log folder: {e}")

            try:
                pathlib.Path(self.log_folder_path + "figures").mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"An error occurred while creating figures folder: {e}")

            try:
                pathlib.Path(self.log_folder_path + "/results_master_lists").mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"An error occurred while creating results_master_lists folder: {e}")

            try:
                pathlib.Path(self.log_folder_path + "/progress_logs").mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"An error occurred while creating progress_logs folder: {e}")

            try:
                pathlib.Path(self.log_folder_path + "/progress_logs_scores").mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"An error occurred while creating progress_logs_scores folder: {e}")

            try:
                pathlib.Path(f"{self.log_folder_path}/" + "/torch").mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"An error occurred while creating torch folder: {e}")

            try:
                pathlib.Path(f"{self.log_folder_path}/" + "/xgb").mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"An error occurred while creating xgb folder: {e}")


            self.model_store_path = f"{self.log_folder_path}/"+"/model_store.json"

            model_directory = {'models':{}}
            jsonString = json.dumps(model_directory)
            if(os.path.exists(self.model_store_path)==False):
                jsonFile = open(self.model_store_path, "w")
                jsonFile.write(jsonString)
                jsonFile.close()
                
                
            