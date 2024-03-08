import tqdm


def get_y_pred_ann_torch_weighting(best, ml_grid_object, valid=False):

    y_test_ann = y_test.copy()

    model_train_time_warning_threshold = 15
    start = time.time()

    target_ensemble = best[0]
    # Get prediction matrix
    # print('Get prediction matrix')
    if valid:
        print(f"get_y_pred_ann_torch_weighting {valid}")
        x_test = X_test_orig.copy()
        y_test_ann = y_test_orig.copy()

        prediction_array = []

        for i in tqdm(range(0, len(target_ensemble))):
            # For model i, predict it's x_test
            feature_columns = list(
                target_ensemble[i][2]
            )  # get features model was trained on

            if type(target_ensemble[i][1]) is not BinaryClassification:

                model = target_ensemble[i][1]

                model.fit(X_train[feature_columns], y_train)

                prediction_array.append(
                    model.predict(X_train[feature_columns])
                )  # Use model to predict x train

            else:

                test_data = TestData(torch.FloatTensor(X_train[feature_columns].values))
                test_loader = DataLoader(dataset=test_data, batch_size=1)

                device = torch.device("cpu")
                model = target_ensemble[i][1]
                model.to(device)  # Has this model been fitted??
                y_hat = model(test_data.X_data.to(device))

                y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

                y_hat = y_hat.astype(int).flatten()
                prediction_array.append(y_hat)

        prediction_matrix_X_train = np.matrix(prediction_array)
        prediction_matrix_X_train = prediction_matrix_X_train.astype(float)
        prediction_matrix_raw_X_train = (
            prediction_matrix_X_train  # Store predictions from x_train into matrix
        )

        X_prediction_matrix_raw_X_train = prediction_matrix_raw_X_train.T

        # Produce test results for valid
        prediction_array = []
        for i in tqdm(range(0, len(target_ensemble))):
            feature_columns = list(target_ensemble[i][2])
            if type(target_ensemble[i][1]) is not BinaryClassification:
                prediction_array.append(
                    target_ensemble[i][1].predict(x_test[feature_columns])
                )  # Generate predictions from stored models on validset

            else:

                test_data = TestData(torch.FloatTensor(x_test[feature_columns].values))
                test_loader = DataLoader(dataset=test_data, batch_size=1)

                device = torch.device("cpu")
                model = target_ensemble[i][1]
                model.to(device)
                y_hat = model(test_data.X_data.to(device))

                y_hat = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()

                y_hat = y_hat.astype(int).flatten()
                prediction_array.append(y_hat)

        prediction_matrix_X_test = np.matrix(prediction_array)
        prediction_matrix_X_test = prediction_matrix_X_test.astype(float)
        prediction_matrix_raw_X_test = prediction_matrix_X_test

        prediction_matrix_raw_X_test = (
            prediction_matrix_raw_X_test.T
        )  # Transpose predictions into columns for each model. X >>y
        test_data = TestData(torch.FloatTensor(prediction_matrix_raw_X_test))

    elif valid == False:
        # Make predictions on xtrain and y train, feed results into nn to learn weights to map ensemble to true. Apply nn to test ensemble preds
        prediction_array = []
        for i in tqdm(range(0, len(target_ensemble))):

            prediction_array.append(
                target_ensemble[i][5]
            )  # Get stored y_pred from x_test (non validation set)

        prediction_matrix_X_train = np.matrix(prediction_array)
        prediction_matrix_X_train = prediction_matrix_X_train.astype(float)
        prediction_matrix_raw_X_train = prediction_matrix_X_train

        X_prediction_matrix_raw_X_train = (
            prediction_matrix_raw_X_train.T
        )  # transpose to matrix, columns are each model yhat vector

        test_data = TestData(
            torch.FloatTensor(X_prediction_matrix_raw_X_train)
        )  # set test data to train set, only learn weights from training
        # y_test = y_train.copy()
        prediction_matrix_raw_X_test = X_prediction_matrix_raw_X_train

    train_data = TrainData(
        torch.FloatTensor(X_prediction_matrix_raw_X_train),
        torch.FloatTensor(np.array(y_train)),
    )  # data set to learn weights for x_train model preds to y_train labels

    # print(len(prediction_array[0]))
    # print(prediction_matrix_raw_X_test.shape)

    y_pred_unweighted = []
    for i in range(0, len(prediction_array[0])):
        y_pred_unweighted.append(round(np.mean(prediction_matrix_raw_X_test.T[:, i])))

    auc = metrics.roc_auc_score(y_test_ann, y_pred_unweighted)

    mccscore_unweighted = matthews_corrcoef(y_test_ann, y_pred_unweighted)

    y_pred_ensemble = train_ann_weight(
        X_prediction_matrix_raw_X_train.shape[1],
        int(X_prediction_matrix_raw_X_train.shape[0]),
        train_data,
        test_data,
    )

    # print("Ensemble ANN weighting training AUC: ", auc_score_weighted)

    if any(np.isnan(y_pred_ensemble)):
        print("Torch model nan, returning random y pred vector")
        # zero_vector = [x for x in range(0, len(y_pred))]
        # y_pred = zero_vector
        random_y_pred_vector = (
            np.random.choice(
                a=[False, True],
                size=(
                    len(
                        y_test_ann,
                    )
                ),
            )
        ).astype(int)
        y_pred = random_y_pred_vector
        y_pred_ensemble = random_y_pred_vector
    else:
        # plot_auc(y_hat, f"Deep ANN Torch {para_str}")
        pass

    auc_score_weighted = metrics.roc_auc_score(y_test_ann, y_pred_ensemble)

    mccscore_weighted = matthews_corrcoef(y_test_ann, y_pred_ensemble)

    auc_score_weighted = round(metrics.roc_auc_score(y_test_ann, y_pred_ensemble), 4)
    print("ANN unweighted ensemble AUC: ", auc)
    print("ANN weighted   ensemble AUC: ", auc_score_weighted)
    print("ANN weighted   ensemble AUC difference: ", auc_score_weighted - auc)
    print("ANN unweighted ensemble MCC: ", mccscore_unweighted)
    print("ANN weighted   ensemble MCC: ", mccscore_weighted)

    # score = (1-de.fun)
    # optimal_weights = de.x

    end = time.time()
    model_train_time = int(end - start)
    #
    # print(len(y_pred_ensemble))
    torch.cuda.empty_cache()  # exp

    return y_pred_ensemble
