import numpy as np
import statsmodels as stats
import torch


def get_best_y_pred_unweighted(best, ml_grid_object, valid=False):

    local_param_dict = ml_grid_object.local_param_dict
    X_test_orig = ml_grid_object.X_test_orig
    y_test_orig = ml_grid_object.y_test_orig

    if valid:
        x_test = X_test_orig.copy()
        y_test = y_test_orig.copy()

    prediction_array = []
    target_ensemble = best[0]
    if valid:
        for i in range(0, len(target_ensemble)):

            feature_columns = list(
                target_ensemble[i][2]
            )  # list(target_ensemble[i][3].columns)

            if type(target_ensemble[i][1]) is not BinaryClassification:

                model = target_ensemble[i][1]

                model.fit(X_train[feature_columns], y_train)

                prediction_array.append(model.predict(x_test[feature_columns]))

            else:

                test_data = TestData(torch.FloatTensor(x_test[feature_columns].values))
                test_loader = DataLoader(dataset=test_data, batch_size=1)

                device = torch.device("cpu")
                model = target_ensemble[i][1]  # Has this model been fitted already?
                model.to(device)

                # model.fit(X_train, y_train)

                y_hat = model(test_data.X_data.to(device))

                y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

                y_hat = y_hat.astype(int).flatten()
                prediction_array.append(y_hat)

    else:
        for i in range(0, len(target_ensemble)):
            y_pred = target_ensemble[i][5]
            prediction_array.append(y_pred)

    prediction_matrix = np.matrix(prediction_array)

    # collapse the mean of each models prediction for each case into a binary label returning a final y_pred composite score from each model
    y_pred_best = []
    for i in range(0, len(prediction_array[0])):
        # y_pred_best.append(round(np.mean(np.matrix(prediction_array)[:,i])))
        try:
            y_pred_best.append(
                stats.mode(np.matrix(prediction_array)[:, i], keepdims=True)[0][0][0]
            )
        except:
            y_pred_best.append(stats.mode(np.matrix(prediction_array)[:, i])[0][0][0])
    return y_pred_best
