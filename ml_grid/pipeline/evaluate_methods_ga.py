import time
from ml_grid.ga_functions.ga_ann_weight_methods import get_y_pred_ann_torch_weighting
from ml_grid.ga_functions.ga_de_weight_method import (
    get_weighted_ensemble_prediction_de_y_pred_valid,
)
from ml_grid.ga_functions.ga_ensemble_weight_finder_de import (
    super_ensemble_weight_finder_differential_evolution,
)

# from ml_grid.ga_functions.ga_ensemble_weight_finder_de import (
#     super_ensemble_weight_finder_differential_evolution,
# )

# from ml_grid.ga_functions.ga_ensemble_weight_finder_de import (
#     super_ensemble_weight_finder_differential_evolution,
# )
from ml_grid.ga_functions.ga_unweighted import get_best_y_pred_unweighted

import numpy
import numpy as np
import pandas as pd
import scipy
import torch
import tqdm
from ml_grid.pipeline import torch_binary_classification_method_ga
from ml_grid.pipeline.torch_binary_classification_method_ga import (
    BinaryClassification,
    TestData,
)

from ml_grid.util.global_params import global_parameters
from sklearn import metrics


# def get_y_pred_resolver(ensemble, ml_grid_object, valid=False):
#     local_param_dict = ml_grid_object.local_param_dict
#     X_test_orig = ml_grid_object.X_test_orig
#     y_test_orig = ml_grid_object.y_test_orig

#     if (
#         local_param_dict.get("weighted") == None
#         or local_param_dict.get("weighted") == "unweighted"
#     ):
#         y_pred = get_best_y_pred_unweighted(ensemble, ml_grid_object, valid=valid)
#     elif local_param_dict.get("weighted") == "de":
#         y_pred = get_weighted_ensemble_prediction_de_y_pred_valid(
#             ensemble,
#             super_ensemble_weight_finder_differential_evolution(
#                 ensemble, ml_grid_object, valid=valid
#             ),
#             valid=valid,
#         )
#         # print(y_pred.shape)
#     elif local_param_dict.get("weighted") == "ann":
#         y_pred = get_y_pred_ann_torch_weighting(ensemble, ml_grid_object, valid=valid)

#     return y_pred


def get_y_pred_resolver(ensemble, ml_grid_object, valid=False):
    print("get_y_pred_resolver")
    print(ensemble)
    local_param_dict = ml_grid_object.local_param_dict
    X_test_orig = ml_grid_object.X_test_orig
    y_test_orig = ml_grid_object.y_test_orig

    if ml_grid_object.verbose >= 2:
        print("Starting get_y_pred_resolver function...")
        print("local_param_dict:", local_param_dict)
        print("X_test_orig shape:", X_test_orig.shape)
        print("y_test_orig shape:", y_test_orig.shape)

    if (
        local_param_dict.get("weighted") == None
        or local_param_dict.get("weighted") == "unweighted"
    ):
        if ml_grid_object.verbose >= 1:
            print("Using unweighted ensemble prediction...")
        y_pred = get_best_y_pred_unweighted(ensemble, ml_grid_object, valid=valid)
    elif local_param_dict.get("weighted") == "de":
        if ml_grid_object.verbose >= 1:
            print("Using DE weighted ensemble prediction...")
        y_pred = get_weighted_ensemble_prediction_de_y_pred_valid(
            ensemble,
            super_ensemble_weight_finder_differential_evolution(
                ensemble, ml_grid_object, valid=valid
            ),
            ml_grid_object,
            valid=valid,
        )
        if ml_grid_object.verbose >= 2:
            print("DE weighted y_pred shape:", y_pred.shape)
    elif local_param_dict.get("weighted") == "ann":
        if ml_grid_object.verbose >= 1:
            print("Using ANN weighted ensemble prediction...")
        y_pred = get_y_pred_ann_torch_weighting(ensemble, ml_grid_object, valid=valid)

    return y_pred


def evaluate_weighted_ensemble_auc(individual, ml_grid_object):

    if ml_grid_object.verbose >= 1:
        print("evaluate_weighted_ensemble_auc: individual")
        print(individual)

    global_params = global_parameters()

    ml_grid_object = ml_grid_object

    verbose = ml_grid_object.global_params.verbose

    y_test = ml_grid_object.y_test

    local_param_dict = ml_grid_object.local_param_dict

    file_path = ml_grid_object.logging_paths_obj.log_folder_path

    orignal_feature_names = ml_grid_object.orignal_feature_names

    log_store_dataframe_path = global_params.log_store_dataframe_path

    mccScoresList = []

    aucScoresList = []

    # print("Evaluating individual of size: ", len(individual[0]))

    # individual_data = individual[0]

    y_pred = get_y_pred_resolver(individual, ml_grid_object, valid=False)

    # print(y_test.shape, y_pred.shape, )
    # should write mcc parallel instead of auc also, need pair functions
    auc = metrics.roc_auc_score(y_test, y_pred)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average="binary")
    precision = metrics.precision_score(y_test, y_pred, average="binary")
    recall = metrics.recall_score(y_test, y_pred, average="binary")
    accuracy = metrics.accuracy_score(y_test, y_pred)

    if verbose >= 1:
        print(f"Ensemble MCC {mcc}, AUC {auc}, nb {len(individual[0])}")

    # ?? how does the diversity weighting and the order of operations interact with the ensemble weights already
    # measure and incorporate diversity
    diversity_metric = measure_binary_vector_diversity(individual)

    diversity_parameter = local_param_dict.get("div_p")
    #     final_score = (auc* diversity_metric * diversity_parameter)/2
    #     final_score_mcc = (mcc* diversity_metric * diversity_parameter)/2

    # Simple diversity incorporation
    auc_div = auc * ((-1 + (diversity_metric + diversity_parameter)))
    # div metric should decrease score as 1=similar, we want to penalise similarity of prediction

    mcc_div = mcc * ((-1 + (diversity_metric + diversity_parameter)))

    # blank init with headers in main log_store_dataframe_path
    ensemble_model_list = []
    feature_count_list = []
    auc_score_list = []
    mcc_score_list = []
    # For each member in the ensemble
    for i in range(0, len(individual[0]) - 1):
        ensemble_model_list.append(str(individual[0][i][1]))  # str or model?
        feature_count_list.append(
            individual[0][i][2]
        )  # could replace with length of list to reduce size on disk
        auc_score_list.append(individual[0][i][4])
        mcc_score_list.append(individual[0][i][0])

    # orignal_feature_names

    feature_map_vector = []
    for col in orignal_feature_names:
        if col in feature_count_list:
            feature_map_vector.append(1)
        else:
            feature_map_vector.append(0)

    feature_map_vector = np.array(feature_map_vector)

    # set score log dataframe in main
    df_data = [
        [
            len(individual[0]),
            auc,
            np.mean(auc_score_list),
            auc_div,
            mcc,
            np.mean(mcc_score_list),
            mcc_div,
            f1,
            precision,
            recall,
            accuracy,
            ensemble_model_list,
            feature_map_vector,
            auc_score_list,
            mcc_score_list,
        ]
    ]
    column_headers = [
        "n",
        "auc",
        "auc_mean",
        "auc_div",
        "mcc",
        "mcc_mean",
        "mcc_div",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "ensemble_model_list",
        "feature_count_list",
        "auc_score_list",
        "mcc_score_list",
    ]

    #     column_headers = ['nb_size', 'f_list', 'auc','mcc','f1','precision','recall','accuracy', 'nb_val', 'pop_val', 'g_val', 'g', 'weighted', 'use_stored_base_learners', 'store_base_learners',
    #        'resample', 'scale', 'n_features', 'param_space_size', 'n_unique_out',
    #        'outcome_var_n', 'div_p', 'percent_missing', 'corr',
    #                        'age', 'sex', 'bmi','ethnicity', 'bloods', 'diagnostic_order',
    #                       'drug_order', 'annotation_n', 'meta_sp_annotation_n',
    #                       'X_train_size', 'X_test_orig_size', 'X_test_size',
    #                    'run_time', 'cxpb', 'mutpb', 'indpb', 't_size']

    df = pd.DataFrame(data=df_data, columns=column_headers)
    df.to_csv(
        f"{file_path}/progress_logs_scores/{log_store_dataframe_path}.csv",
        mode="a",
        index=True,
        header=False,
    )

    if verbose >= 1:
        print(
            f"""Ensemble MCC {mcc}, diversity weighted MCC {mcc_div}
        , \n f1 {f1} precision {precision} recall {recall} accuracy {accuracy}
        , \n AUC {auc}, diversity weighted AUC {auc_div}
        , \n nb {len(individual[0])}, diversity_score: {diversity_metric}, diff: {auc_div-auc} """
        )

    # return mcc for genetic algorithm evalutation
    if diversity_parameter > 0:
        return (auc_div,)
    else:
        return (auc,)
    # return (auc,)


# redundant? weights only derived from xtrain, weight vec is size of ensemble not train set

# Only get weights from xtrain/ytrain, never get weights from xtest y test. Use weights on x_validation yhat to compare to ytrue_valid


# %%cython -a

import numpy as np
from numpy.linalg import norm
from sklearn import metrics

round_v = np.vectorize(round)


def normalize(weights):
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result


# def get_weighted_ensemble_prediction_de_cython(weights, prediction_matrix_raw, y_test):
#     """Function used by DE algo to search for optimal weights with scoring"""

#     clean_prediction_matrix = prediction_matrix_raw.copy()
#     weights = normalize(weights)

#     weighted_prediction_matrix_array = (
#         np.array(clean_prediction_matrix) * weights[:, None]
#     )
#     collapsed_weighted_prediction_matrix_array = weighted_prediction_matrix_array.sum(
#         axis=0
#     )

#     y_pred_best = round_v(collapsed_weighted_prediction_matrix_array)

#     auc = metrics.roc_auc_score(y_test, y_pred_best)
#     score = auc

#     # mcc = metrics.matthews_corrcoef(y_test, y_pred_best)

#     return 1 - score


def measure_binary_vector_diversity(ensemble, metric="jaccard"):
    # beta
    # can select any from scipy spatial distance pdist
    # if two datasets have a Jaccard Similarity of 80% then they would have a Jaccard distance of 1 – 0.8 = 0.2 or 20%
    # closer to zero, the more similar the vectors
    # if all are similar the less diverse they are
    ##assume mcc is 0.5, n = 2
    # then (0.5 * 0.1)*2  = 0.1
    # and  (0.5 * 0.5)*2) = 0.5
    # needs diversity_parameter to modify strength?

    n_y_pred = len(ensemble[0])  # check level

    all_y_pred_arrays = []

    for i in range(0, n_y_pred):
        all_y_pred_arrays.append(ensemble[0][i][5])

    distance_vector = scipy.spatial.distance.pdist(all_y_pred_arrays, metric=metric)

    return np.mean(distance_vector)
