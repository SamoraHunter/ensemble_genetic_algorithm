import json
import logging
import os
import pathlib
from typing import Dict, Optional


class log_folder:
    """Manages the creation of a structured logging directory for an experiment.

    This class generates a unique folder name based on the hyperparameters of a
    given run. It then creates a standard set of subdirectories within this
    folder (e.g., for logs, figures, models) to keep all experiment artifacts
    organized.
    """

    additional_naming: Optional[str]
    """An optional string to append to the main log folder name."""

    base_project_dir: str
    """The root directory where the log folders will be created."""

    global_param_str: str
    """A condensed string representation of the run's hyperparameters, used for the folder name."""

    log_folder_path: str
    """The full path to the main logging directory for the current run."""

    logging_path: str
    """The full path to the primary text log file (`logging.txt`)."""

    model_store_path: str
    """The full path to the JSON file used for storing model metadata."""

    def __init__(
        self,
        local_param_dict: Dict,
        additional_naming: Optional[str],
        base_project_dir: str,
    ):
        """Initializes the log_folder object and creates the directory structure.

        This constructor takes the parameters for a specific run, converts them
        into a unique, shortened string, and uses that string to create a
        main folder for the run's logs and artifacts. It then populates this
        folder with a standard set of subdirectories.

        Args:
            local_param_dict: A dictionary of parameters for the specific run.
            additional_naming: An optional string to append to the folder name
                for easier identification.
            base_project_dir: The root directory for the project where all log
                folders will be stored.
        """
        self.additional_naming = additional_naming
        self.base_project_dir = base_project_dir

        str_b = ""
        for key in local_param_dict.keys():
            if key != "data":
                str_b = str_b + "_" + str(local_param_dict.get(key))
            else:
                for key in local_param_dict.get("data"):
                    str_b = str_b + str(int(local_param_dict.get("data").get(key)))

        self.global_param_str = str_b
        # self.global_param_str = str(global_param_dict).replace("{", "").replace("}", "").replace(":", "").replace(" ", "").replace(",", "").replace("'", "_").replace("__", "_").replace("'","").replace(",","").replace(": ", "_").replace("{","").replace("}","").replace("True","T").replace("False", "F").replace(" ","_").replace("[", "").replace("]", "").replace("_","")

        print(self.global_param_str)
        words = self.global_param_str.split(
            "_"
        )  # Split the string into words using underscores
        for i in range(len(words)):
            if words[i] == "True":
                words[i] = "T"
            elif words[i] == "False":
                words[i] = "F"
            elif len(words[i]) > 0:  # Replace full word with its first letter
                try:
                    float_val = float(words[i])
                    if float_val.is_integer():
                        words[i] = str(int(float_val))
                    else:
                        words[i] = str(float_val).lstrip("0")
                except ValueError:
                    words[i] = words[i][0]

        self.global_param_str = "_".join(words)
        print(self.global_param_str)

        # self.global_param_str = "test_dir"

        # self.log_folder_path = f"{self.global_param_str + additional_naming}/logs/"

        # pathlib.Path(self.base_project_dir + self.log_folder_path).mkdir(
        #     parents=True, exist_ok=True
        # )

        # full_log_path = f"{self.base_project_dir+self.global_param_str + additional_naming}/logs/log.log"

        full_log_path = (
            f"{self.base_project_dir+self.global_param_str + additional_naming}/logs/"
        )

        # full_log_path = ".log/"

        self.log_folder_path = full_log_path

        def ensure_full_log_path(full_log_path):
            directory = os.path.dirname(full_log_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

        ensure_full_log_path(full_log_path)

        try:
            # Check if the directory exists, if not, create it.
            log_dir = os.path.dirname(full_log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            self.logging_path = os.path.join(full_log_path, "logging.txt")

            # Check if the file exists, if not, create it.
            if not os.path.exists(self.logging_path):
                with open(self.logging_path, "w"):
                    pass  # Create the file

            # Set up logging
            logging.basicConfig(filename=self.logging_path)
            stderrLogger = logging.StreamHandler()
            stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
            logging.getLogger().addHandler(stderrLogger)

            # Test logging
            logging.info("Logging setup successful")

        except Exception as e:
            print("Failed to set log dir at ", self.logging_path)
            print(e)

        try:
            logging.basicConfig(filename=full_log_path)
            stderrLogger = logging.StreamHandler()
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
            pathlib.Path(self.log_folder_path + "figures").mkdir(
                parents=True, exist_ok=True
            )
        except Exception as e:
            print(f"An error occurred while creating figures folder: {e}")

        try:
            pathlib.Path(self.log_folder_path + "/results_master_lists").mkdir(
                parents=True, exist_ok=True
            )
        except Exception as e:
            print(f"An error occurred while creating results_master_lists folder: {e}")

        try:
            pathlib.Path(self.log_folder_path + "/progress_logs").mkdir(
                parents=True, exist_ok=True
            )
        except Exception as e:
            print(f"An error occurred while creating progress_logs folder: {e}")

        try:
            pathlib.Path(self.log_folder_path + "/progress_logs_scores").mkdir(
                parents=True, exist_ok=True
            )
        except Exception as e:
            print(f"An error occurred while creating progress_logs_scores folder: {e}")

        try:
            pathlib.Path(f"{self.log_folder_path}/" + "/torch").mkdir(
                parents=True, exist_ok=True
            )
        except Exception as e:
            print(f"An error occurred while creating torch folder: {e}")

        try:
            pathlib.Path(f"{self.log_folder_path}/" + "/xgb").mkdir(
                parents=True, exist_ok=True
            )
        except Exception as e:
            print(f"An error occurred while creating xgb folder: {e}")

        self.model_store_path = f"{self.log_folder_path}/" + "/model_store.json"

        model_directory = {"models": {}}
        jsonString = json.dumps(model_directory)
        if os.path.exists(self.model_store_path) == False:
            jsonFile = open(self.model_store_path, "w")
            jsonFile.write(jsonString)
            jsonFile.close()
