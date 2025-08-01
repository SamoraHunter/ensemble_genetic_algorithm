import numpy as np
import torch
import numpy
from torch.utils.data import Dataset, DataLoader
from numpy.linalg import norm
import torch.nn as nn
from ml_grid.ga_functions.ga_ann_util import BinaryClassification, TestData, normalize


def get_weighted_ensemble_prediction_de_y_pred_valid(
    best, weights, ml_grid_object, valid=False
):
    """
    Generates weighted ensemble predictions for a given set of models and weights, either on the validation set or using precomputed predictions.
    Args:
        best (list): List containing the ensemble configuration. The first element should be a list of tuples, each representing a model and its associated metadata.
        weights (array-like): Array of weights to apply to each model's predictions in the ensemble.
        ml_grid_object (object): An object containing training and test data, as well as other configuration parameters. Must have attributes such as X_train, X_test, y_train, X_test_orig, y_test_orig, and verbose.
        valid (bool, optional): If True, computes predictions on the validation set using the models. If False, uses precomputed predictions from the ensemble configuration. Defaults to False.
    Returns:
        numpy.ndarray: The final weighted ensemble predictions, rounded to the nearest integer (typically 0 or 1 for classification tasks).
    """

    round_v = np.vectorize(round)

    X_test_orig = ml_grid_object.X_test_orig
    y_test_orig = ml_grid_object.y_test_orig
    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train

    target_ensemble = best[0]
    if valid:
        if ml_grid_object.verbose >= 1:
            print("Evaluating weighted ensemble on validation set")
        x_test = X_test_orig.copy()
        y_test = y_test_orig.copy()

        prediction_array = []

        for i in range(0, len(target_ensemble)):
            feature_columns = target_ensemble[i][2]

            model = target_ensemble[i][1]

            if type(target_ensemble[i][1]) is not BinaryClassification:
                if ml_grid_object.verbose >= 2:
                    print(f"Fitting model {i+1}")

                try:
                    model.fit(X_train[feature_columns], y_train)
                except ValueError as e:
                    print(e)
                    print("ValueError on fit")
                    print("feature_columns")
                    print(len(feature_columns))
                    print(
                        X_train.shape,
                        X_test.shape,
                        type(X_train),
                        type(y_train),
                        type(feature_columns),
                    )
                    print(
                        "Warning: The following columns do not exist in feature_columns:"
                    )
                    print("\n".join(missing_columns))
                    print(print(len(missing_columns), len(feature_columns)))

                prediction_array.append(model.predict(x_test[feature_columns]))
            else:
                if ml_grid_object.verbose >= 2:
                    print(f"Handling torch model prediction for model {i+1}")
                test_data = TestData(torch.FloatTensor(x_test[feature_columns].values))
                test_loader = DataLoader(dataset=test_data, batch_size=1)

                device = torch.device("cpu")
                model.to(device)
                y_hat = model(test_data.X_data.to(device))

                y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

                y_hat = y_hat.astype(int).flatten()

                if numpy.isnan(y_hat).any():
                    print(
                        "Returning dummy random yhat vector for torch pred, nan found"
                    )
                    y_hat = np.random.choice(a=[False, True], size=(len(y_hat),))

                prediction_array.append(y_hat)
    else:
        prediction_array = []

        for i in range(0, len(target_ensemble)):
            prediction_array.append(target_ensemble[i][5])

    prediction_matrix = np.matrix(prediction_array)
    prediction_matrix = prediction_matrix.astype(float)
    prediction_matrix_raw = prediction_matrix

    clean_prediction_matrix = prediction_matrix_raw.copy()
    weights = normalize(weights)

    weighted_prediction_matrix_array = (
        np.array(clean_prediction_matrix) * weights[:, None]
    )
    collapsed_weighted_prediction_matrix_array = weighted_prediction_matrix_array.sum(
        axis=0
    )

    y_pred_weighted = round_v(collapsed_weighted_prediction_matrix_array)

    torch.cuda.empty_cache()  # exp

    return y_pred_weighted
