import json
import pickle
import random
import time
import logging

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
logger = logging.getLogger("ensemble_ga")


def store_model(
    ml_grid_object: Any,
    local_param_dict: Dict,
    mccscore: float,
    model: Any,
    feature_list: List[str],
    model_train_time: int,
    auc_score: float,
    y_pred: np.ndarray,
    model_type: str = "sklearn",
) -> None:
    """Stores a trained model's metadata and object to disk.

    This function serializes a model and its performance metrics into a central
    JSON file (`model_store.json`). It handles different model types:
    - `sklearn`: The model object is converted to its string representation.
    - `torch`: The model is saved to a separate file using `torch.save`, and a
      timestamp is stored in the JSON.
    - `xgb`: The model is pickled to a separate file, and a timestamp is stored.

    Args:
        ml_grid_object: An object containing project configurations, including
            logging paths.
        local_param_dict: A dictionary of local parameters for the run.
        mccscore: The Matthews Correlation Coefficient score of the model.
        model: The trained model object to be stored.
        feature_list: A list of feature names used by the model.
        model_train_time: The time taken to train the model, in seconds.
        auc_score: The ROC AUC score of the model.
        y_pred: The model's predictions on the test set.
        model_type: The type of the model ('sklearn', 'torch', 'xgb').
            Defaults to "sklearn".
    """
    if ml_grid_object.verbose >= 11:
        logger.debug("store_model")

    model_store_path = ml_grid_object.logging_paths_obj.model_store_path

    global_param_str = ml_grid_object.logging_paths_obj.global_param_str

    additional_naming = ml_grid_object.logging_paths_obj.additional_naming

    global_param_dict = ml_grid_object.global_params

    log_folder_path = ml_grid_object.logging_paths_obj.log_folder_path

    if ml_grid_object.verbose >= 1:
        logger.info("model_store_path: %s", model_store_path)
        logger.info("log_folder_path: %s", log_folder_path)

    with open(model_store_path, "r") as f:
        model_store_data = json.load(f)

    idx = len(model_store_data["models"]) + 1

    time_stamp = time.time_ns()

    if ml_grid_object.verbose >= 11:
        logger.debug("saving model type: %s", model_type)

    if model_type == "sklearn":
        model = str(model)

    elif model_type == "torch":
        y_pred = y_pred.astype(float)
        torch.save(model, f=f"{log_folder_path}/" + "/torch/" + str(time_stamp))
        model = time_stamp

    elif model_type == "xgb":
        pickle.dump(
            model, open(f"{log_folder_path}/" + "/xgb/" + str(time_stamp), "wb")
        )
        model = time_stamp
        y_pred = y_pred.astype(float)

    # print(type(model))
    scale = local_param_dict.get("scale")
    if scale:
        y_pred = y_pred.astype(float)

    model_store_entry = {
        "index": idx,
        "mcc_score": mccscore,
        "model": model,
        "feature_list": feature_list,
        "model_train_time": model_train_time,
        "auc_score": auc_score,
        "y_pred": list(y_pred),
        "model_type": model_type,
    }

    model_store_data["models"].update({idx: model_store_entry})

    jsonString = json.dumps(model_store_data)
    jsonFile = open(model_store_path, "w", encoding="utf-8")
    jsonFile.write(jsonString)
    jsonFile.close()

    try:
        torch.cuda.empty_cache()  # exp
    except Exception as e:
        logger.warning("Failed to torch empty cache: %s", e)


def get_stored_model(ml_grid_object: Any) -> Tuple:
    """Retrieves a randomly selected, previously stored model.

    This function reads the `model_store.json` file, randomly picks one of
    the stored models, and deserializes it. It handles different model types:
    - `sklearn`: Re-creates the model object using `eval()`.
    - `torch`: Loads the model from its file using `torch.load()`.
    - `xgb`: Unpickles the model from its file.

    If retrieving a stored model fails for any reason, it falls back to
    generating a new random model using the `modelFuncList`.

    Args:
        ml_grid_object: An object containing project configurations, including
            logging paths and the `modelFuncList`.

    Returns:
        A tuple containing the model's performance and objects, in the format:
        (mccscore, model, feature_list, model_train_time, auc_score, y_pred).

    Warning:
        This function uses `eval()` to reconstruct scikit-learn models from
        their string representation. This can be a security risk if the
        `model_store.json` file is from an untrusted source.
    """
    model_store_path = ml_grid_object.logging_paths_obj.model_store_path

    global_param_str = ml_grid_object.logging_paths_obj.global_param_str

    additional_naming = ml_grid_object.logging_paths_obj.additional_naming

    global_param_dict = ml_grid_object.global_params

    log_folder_path = ml_grid_object.logging_paths_obj.log_folder_path

    modelFuncList = ml_grid_object.config_dict.modelFuncList

    with open(model_store_path, "r", encoding="utf-8") as f:
        model_store_data = json.load(f)

    model_key_list = list(model_store_data["models"].keys())

    try:
        model_key = str(random.choice(model_key_list))

        logger.info("Returning stored model at index %s/%s", model_key, len(model_key_list))

        if model_store_data["models"].get(model_key)["model_type"] == "sklearn":
            model = eval(model_store_data["models"].get(model_key)["model"])

        elif model_store_data["models"].get(model_key)["model_type"] == "torch":
            time_stamp = model_store_data["models"].get(model_key)["model"]
            model = torch.load(f=f"{log_folder_path}/" + "/torch/" + str(time_stamp))

        elif model_store_data["models"].get(model_key)["model_type"] == "xgb":
            time_stamp = model_store_data["models"].get(model_key)["model"]
            model = pickle.load(
                open(f"{log_folder_path}/" + "/xgb/" + str(time_stamp), "rb")
            )

        return (
            model_store_data["models"].get(model_key)["mcc_score"],
            model,
            model_store_data["models"].get(model_key)["feature_list"],
            model_store_data["models"].get(model_key)["model_train_time"],
            model_store_data["models"].get(model_key)["auc_score"],
            np.array(model_store_data["models"].get(model_key)["y_pred"]),
        )
    except Exception as e:
        logger.error("Failed inside getting stored model, returning random new model: %s", e)
        index = random.randint(0, len(modelFuncList) - 1)

        return modelFuncList[index](ml_grid_object, ml_grid_object.local_param_dict)
