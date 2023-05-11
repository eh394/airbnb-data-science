import datetime
import itertools
import json
import numpy as np
import os
import torch
import yaml
from torcheval.metrics.functional import r2_score
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import data_config, model_config
from lib import data_utils

np.random.seed(2)


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


def get_nn_config(filename):
    with open(filename, "r") as params:
        params = yaml.safe_load(params)
        return params


def train(model, data_loader, params, epochs=10):

    optimiser = getattr(torch.optim, params["optimiser"])
    optimiser = optimiser(model.parameters(), lr=params["lr"])
    loss_fn = F.mse_loss
    writer = SummaryWriter()
    batch_idx = 0

    for epoch in range(epochs):
        for batch in data_loader:
            X, y = batch
            yhat = model(X)
            loss = loss_fn(yhat, y)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar("loss", loss.item(), batch_idx)
            batch_idx += 1

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
    for params in permutations:
        metrics = {}
        model = NN(params)
        start = datetime.datetime.now()
        model = train(model, train_loader, params)
        metrics["training_duration"] = (
            datetime.datetime.now() - start).microseconds
        metrics["RMSE_train"], metrics["R2_train"], latency_train_ = evaluate_model(
            model, X_train, y_train)
        metrics["RMSE_validation"], metrics["R2_validation"], metrics["inference_latency"] = evaluate_model(
            model, X_validation, y_validation)
        summary_params_metrics.append([params, metrics])
    return summary_params_metrics


def save_model(trained_model, output_folder, opt_hyperparams=None, metrics=None):
    now = datetime.datetime.now()
    output_folder = f"{output_folder}/{now}"
    os.makedirs(output_folder)
    filepath = f"{output_folder}/model.pt"
    torch.save(trained_model.state_dict(), filepath)

    filepath = f"{output_folder}/hyperparameters.json"
    json.dump(opt_hyperparams, open(filepath, "w"))

    filepath = f"{output_folder}/metrics.json"
    json.dump(metrics, open(filepath, "w"))


def find_best_nn(summary):
    RMSE_opt = summary[0][1]["RMSE_validation"]
    params_opt = {}
    metrics_opt = {}
    for entry in summary:
        if entry[1]["RMSE_validation"] < RMSE_opt:
            RMSE_opt = entry[1]["RMSE_validation"]
            params_opt = entry[0]
            metrics_opt = entry[1]

    model = NN(params_opt)
    save_model(
        model,
        output_folder="models/neural_networks/regression/",
        opt_hyperparams=params_opt,
        metrics=metrics_opt
    )

    return model, params_opt, metrics_opt


if __name__ == "__main__":

    df = data_utils.load_df(
        raw_data_filename="listings.csv",
        clean_data_filename="listings_clean.csv",
        missing_values_subset=data_config.rating_columns,
        description_string_subset="Description",
        default_values_subset=data_config.default_value_columns,
        default_value=1
    )

    X_train, y_train, X_validation, y_validation, X_test, y_test = data_utils.load_split_X_y(
        df,
        features=data_config.feature_columns,
        labels="Price_Night",
        train_test_proportion=0.7,
        test_validation_proportion=0.5
    )

    train_data = X_train, y_train
    validation_data = X_validation, y_validation
    test_data = X_test, y_test

    train_dataset = AirbnbNightlyPriceRegressionDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    validation_dataset = AirbnbNightlyPriceRegressionDataset(validation_data)
    validation_loader = DataLoader(validation_dataset)
    model, params_opt, metrics_opt = find_best_nn(optimize_nn_params(
        generate_nn_configs(model_config.NN_params)))

    print(params_opt, metrics_opt)
