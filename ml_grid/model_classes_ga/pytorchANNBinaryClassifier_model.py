import itertools
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn import metrics
from ml_grid.ga_functions.ga_ann_util import (
    BinaryClassification,
    TestData,
    TrainData,
    binary_acc,
    get_free_gpu,
)
from ml_grid.util.debug_methods_ga import debug_base_learner
from ml_grid.util.get_feature_selection_class_ga import feature_selection_methods_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.model_methods_ga import store_model
from sklearn.preprocessing import StandardScaler

from ml_grid.util.validate_param_methods import validate_batch_size


def predict_with_fallback(model, X_batch):
    
    print("predict_with_fallback", X_batch.shape)
    
    try:
        y_pred = model(X_batch)
        return y_pred
    except:
        # If an exception occurs (e.g., model prediction fails), generate a random binary vector
        #random_binary_vector = np.random.randint(2, size=X_batch.shape[0])
        X_batch_shape_0 = X_batch.size(0)
        random_binary_vector = torch.randint(2, size=(X_batch_shape_0,))
        
        X_batch_shape = X_batch.shape
        random_binary_tensor = torch.randint(2, size=X_batch_shape)
        print("Returning random_binary_tensor", random_binary_tensor)
        return random_binary_tensor

def Pytorch_binary_class_ModelGenerator(ml_grid_object, local_param_dict):
    global_parameter_val = global_parameters()

    verbose = global_parameter_val.verbose
    store_base_learners = ml_grid_object.global_params.store_base_learners
    scale = ml_grid_object.local_param_dict.get("scale")

    X_train = ml_grid_object.X_train
    X_test = ml_grid_object.X_test
    y_train = ml_grid_object.y_train
    y_test = ml_grid_object.y_test

    debug = ml_grid_object.verbose >= 1

    start = time.time()

    X_train, X_test = feature_selection_methods_class(
        ml_grid_object
    ).get_featured_selected_training_data(method="anova")

    if scale == False:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    train_data = TrainData(
        torch.FloatTensor(X_train.to_numpy()), torch.FloatTensor(y_train.to_numpy())
    )
    test_data = TestData(torch.FloatTensor(X_test.to_numpy()))

    # Initialise global parameter space----------------------------------------------------------------

    # print("ANN binary Xtrain")
    # print(X_train)
    # print(type(X_train))
    # print(int(X_train.shape[0]))

    parameter_space = {
        "column_length": [len(X_train.columns)],
        #'epochs': [50, 200],
        "batch_size": [
            max(2, int(X_train.shape[0] / 100)),
            max(2, int(X_train.shape[0] / 200)),
        ],
        #'learning_rate': lr_space,
        #'learning_rate': [0.1, 0.001, 0.0005, 0.0001],
        "deep_layers_1": [2, 4, 8, 16, 32],
        "dropout_val": [0.1, 0.01, 0.001],
    }

    additional_grid = {
        "epochs": [10, 50, 100],
        "learning_rate": [0.1, 0.001, 0.0005, 0.0001],
    }
    size_test = []
    # Loop over al grid search combinations
    for values in itertools.product(*additional_grid.values()):
        point = dict(zip(additional_grid.keys(), values))
        # merge the general settings
        settings = {**point}
        # print(settings)
        size_test.append(settings)

    # print(len(size_test))

    # Select a random sample from the global parameter space
    sample_parameter_space = {}
    for key in parameter_space.keys():
        sample_parameter_space[key] = random.choice(parameter_space.get(key))

    sample_parameter_space = validate_batch_size(sample_parameter_space)

    additional_param_sample = random.choice(size_test)

    additional_param_sample = {}
    for key in additional_grid.keys():
        additional_param_sample[key] = random.choice(additional_grid.get(key))

    if ml_grid_object.verbose > 0:
        print(sample_parameter_space)

        print(additional_param_sample)

    free_gpu = str(get_free_gpu(ml_grid_object))

    # os.environ["CUDA_VISIBLE_DEVICES"]=free_gpu

    device = torch.device(f"cuda:{free_gpu}" if torch.cuda.is_available() else "cpu")
    if ml_grid_object.verbose > 0:
        print(device)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=sample_parameter_space["batch_size"],
        shuffle=True,
    )
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    # fit model with random sample of global parameter space
    model = BinaryClassification(**sample_parameter_space)

    model.to(device)
    # print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=additional_param_sample["learning_rate"]
    )
    model.train()
    for e in range(1, additional_param_sample["epochs"] + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            #y_pred = model(X_batch)
            #print(X_batch.shape)
            #print(type(X_batch)) #torch torch.Tensor
            try:
                y_pred = predict_with_fallback(model = model, X_batch = X_batch)

            except Exception as e:
                print(e)
                print("Failed ypred fallback")
                print("X_batch shape,", X_batch.shape)
                print("Y_batch.shape", y_batch.shape)
                print("Y_pred.shape", y_pred.shape, type(y_pred), )
                raise e
            
            print("pre: loss = criterion")
            print("X_batch shape,", X_batch.shape)
            print("Y_batch.shape", y_batch.shape)
            print("Y_pred.shape", y_pred.shape, type(y_pred), )

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    if ml_grid_object.verbose > 2:
        print(
            f"Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}"
        )

    para_str = (
        str(settings)
        .replace("'", "_")
        .replace(":", "_")
        .replace(",", "_")
        .replace("{", "_")
        .replace("}", "_")
        .replace(" ", "_")
    ).replace("__", "_")

    try:
        y_pred = model(test_data.X_data.to(device))

        y_pred = torch.round(torch.sigmoid(y_pred)).cpu().detach().numpy().flatten()

    except ValueError as e:
        if verbose >= 1:
            print(e)
            print("Returning random label vector")
            X_test_length = len(X_test)

            y_pred = np.random.randint(2, size=X_test_length)

    if any(np.isnan(y_pred)):
        if ml_grid_object.verbose > 1:
            print("Torch model nan, returning random y pred vector")
        # zero_vector = [x for x in range(0, len(y_pred))]
        # y_pred = zero_vector
        random_y_pred_vector = (
            np.random.choice(
                a=[False, True],
                size=(
                    len(
                        y_test,
                    )
                ),
            )
        ).astype(int)
        y_pred = random_y_pred_vector
    else:
        # plot_auc(y_hat, f"Deep ANN Torch {para_str}")
        pass

    mccscore = matthews_corrcoef(y_test, y_pred)

    auc_score = round(metrics.roc_auc_score(y_test, y_pred), 4)

    end = time.time()
    model_train_time = int(end - start)
    if debug:
        debug_base_learner(model, mccscore, X_train, auc_score, model_train_time)

    if store_base_learners:
        try:
            store_model(
                ml_grid_object,
                local_param_dict,
                mccscore,
                model,
                list(X_train.columns),
                model_train_time,
                auc_score,
                y_pred,
                model_type="torch",
            )
        except Exception as e:
            print(e)

    torch.cuda.empty_cache()  # exp

    return (mccscore, model, list(X_train.columns), model_train_time, auc_score, y_pred)
