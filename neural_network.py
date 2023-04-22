import numpy as np
import pandas as pd
import yaml
import json
import datetime
import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter # figure out how to use this
from torcheval.metrics.functional import r2_score
from sklearn.model_selection import train_test_split
import itertools

import utils_nn
from tabular_data import load_airbnb, rating_columns, default_value_columns

np.random.seed(2)


# Import data - to be moved to / dealt with in another file, possibly utils as can be used in all analysis files
df = pd.read_csv('clean_tabular_data.csv')

features, labels = load_airbnb(
    df, (rating_columns + default_value_columns), 'Price_Night')

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, train_size=0.7, test_size=0.3)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_test, y_test, test_size=0.5)

train_data = X_train, y_train
validation_data = X_validation, y_validation
test_data = X_test, y_test


class AirbnbNightlyPriceRegressionDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        X, y = data
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(-1)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


# Get configuration parameters - move to another file
def get_nn_config(filename):
    with open(filename, "r") as params:
        params = yaml.safe_load(params)
        return params


# params = get_nn_config("nn_config.yaml")


# Function that trains the model
def train(model, data_loader, params, epochs=10):

    optimiser = getattr(torch.optim, params["optimiser"])
    optimiser = optimiser(model.parameters(), lr=params["lr"])
    loss_fn = F.mse_loss
    # writer = SummaryWriter()
    batch_idx = 0

    for epoch in range(epochs):
        for batch in data_loader:  # fix this, makes sense to pass train loader here
            X, y = batch
            yhat = model(X)
            loss = loss_fn(yhat, y)
            loss.backward()
            # print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
            # writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1
        # print(epoch, loss)

    return model


class NN(torch.nn.Module):
    def __init__(self, hyperparams):
        super().__init__()

        n = hyperparams["model_depth"] - 2

        layers = [torch.nn.Linear(
            10, hyperparams["hidden_layer_width"]), torch.nn.ReLU()]
        for i in range(n):
            layers.append(torch.nn.Linear(
                hyperparams["hidden_layer_width"], hyperparams["hidden_layer_width"]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hyperparams["hidden_layer_width"], 1))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)

# this can be a generic function moved to a utils file or similar
def save_model(trained_model, folder, opt_hyperparams=None, metrics=None):

    if str(type(trained_model)) == "<class '__main__.NN'>":
        now = datetime.datetime.now()
        folder = folder + f"{now}"
        os.makedirs(folder)
        filepath = folder + f"/model.pt"
        torch.save(trained_model.state_dict(), filepath)

    else:
        filepath = folder + 'classification_model.joblib'
        joblib.dump(trained_model, filepath)

    filepath = folder + '/hyperparameters.json'
    json.dump(opt_hyperparams, open(filepath, 'w'))

    filepath = folder + '/metrics.json'
    json.dump(metrics, open(filepath, 'w'))


def evaluate_model(model, X, y):

    model.eval()
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(-1)
    start = datetime.datetime.now()
    yhat = model(X)
    inference_latency = (datetime.datetime.now() - start).microseconds / len(X)
    RMSE = torch.sqrt(F.mse_loss(yhat, y)).detach().numpy().item()
    R2_score = r2_score(yhat, y).detach().numpy().item()
    return (RMSE, R2_score, inference_latency)


def generate_nn_configs(params):
    permutations = itertools.product(*list(params.values()))
    accums = []
    for permutation in permutations:
        acc = {key: value for key, value in zip(
            list(params.keys()), permutation
        )}
        accums.append(acc)
    return accums


def optimize_nn_params(permutations):
    summary_params_metrics = []
    run = 1
    for params in permutations:
        metrics = {}
        model = NN(params)
        start = datetime.datetime.now()
        model = train(model, train_loader, params)
        metrics["training_duration"] = (
            datetime.datetime.now() - start).microseconds
        metrics["RMSE_train"], metrics["R2_train"], latency_train = evaluate_model(
            model, X_train, y_train)
        metrics["RMSE_validation"], metrics["R2_validation"], metrics["inference_latency"] = evaluate_model(
            model, X_validation, y_validation)
        run += 1
        summary_params_metrics.append([params, metrics])
    return summary_params_metrics


def find_best_nn(summary):
    RMSE_opt = 1000
    params_opt = {}
    metrics_opt = {}
    for entry in summary:
        # print(entry[1]["RMSE_validation"])
        if entry[1]["RMSE_validation"] < RMSE_opt:
            RMSE_opt = entry[1]["RMSE_validation"]
            params_opt = entry[0]
            metrics_opt = entry[1]

    print(params_opt, metrics_opt)
    model = NN(params_opt)
    save_model(model, "models/neural_networks/regression/",
               opt_hyperparams=params_opt, metrics=metrics_opt)

    return model, params_opt, metrics_opt


if __name__ == "__main__":

    train_dataset = AirbnbNightlyPriceRegressionDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    validation_dataset = AirbnbNightlyPriceRegressionDataset(validation_data)
    validation_loader = DataLoader(validation_dataset)
    find_best_nn(optimize_nn_params(generate_nn_configs(utils_nn.nn_params)))


# state_dict = torch.load('models/neural_networks/regression/2023-04-22 19:40:12.264668/model.pt')
# loaded_model = NN()
# loaded_model.load_state_dict(state_dict)
