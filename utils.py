import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, precision_score, recall_score, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import math
import itertools
import joblib
import json
import os
import datetime
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  # figure out how to use this
from torcheval.metrics.functional import r2_score
from sklearn.model_selection import train_test_split
import itertools
import config
import csv
import os

# Tidy up all imports
# items to remember: changed inputs, changed function name

regression_metrics = ['R_sq', 'RMSE']
classification_metrics = ['accuracy', 'precision', 'recall', 'f1']


def custom_tune_model_params(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        params,
        eval_metrics
):

    acc_metrics = []

    configs = itertools.product(*list(params.values()))
    for config in configs:
        acc = {key: value for key, value in zip(
            list(params.keys()), config)}
        print("foo",model)
        m = model(**acc)
        m.fit(X_train, y_train)

        acc = eval_metrics(m, X_validation, y_validation, acc)
        acc_metrics.append(acc)

    return acc_metrics


def eval_metrics_regression(model, X_validation, y_validation, acc):
    acc['R_sq'] = model.score(X_validation, y_validation)
    acc['RMSE'] = math.sqrt(mean_squared_error(
        y_validation, model.predict(X_validation)))
    return acc


def eval_metrics_classification(model, X_validation, y_validation, acc):

    acc['accuracy'] = accuracy_score(
        y_validation, model.predict(X_validation))
    acc['precision'] = precision_score(
        y_validation, model.predict(X_validation), average='weighted')
    acc['recall'] = recall_score(
        y_validation, model.predict(X_validation), average='weighted')
    acc['f1'] = f1_score(y_validation, model.predict(
        X_validation), average='weighted')
    return acc


def choose_optimal_params(acc_metrics, config_metric):

    assert (config_metric in regression_metrics) or (config_metric in classification_metrics)
    if config_metric in minimise_metrics:
        optim_params = sorted(
            acc_metrics, key=lambda m: m[config_metric], reverse=False)[0]
    elif config_metric in maximise_metrics:
        optim_params = sorted(
            acc_metrics, key=lambda m: m[config_metric], reverse=True)[0]
    return optim_params


def save_model(trained_model, folder, opt_hyperparams=None, metrics=None):

    # if isinstance(trained_model, NN) == True:
    if False == True:
        now = datetime.datetime.now()
        folder = folder + f"{now}"
        os.makedirs(folder)
        filepath = folder + f"/model.pt"
        torch.save(trained_model.state_dict(), filepath)

    else:
        filepath = folder + 'model.joblib'
        joblib.dump(trained_model, filepath)

    filepath = folder + '/hyperparameters.json'
    json.dump(opt_hyperparams, open(filepath, 'w'))

    filepath = folder + '/metrics.json'
    json.dump(metrics, open(filepath, 'w'))


def evaluate_all_models(
        models,
    X_train,
    y_train,
    X_validation,
    y_validation,
    metrics,
    eval_metrics,
    folder,
    config_metric
):

    for model, params in models:

        optim_params = choose_optimal_params(
            custom_tune_model_params(
                model,
                X_train,
                y_train,
                X_validation,
                y_validation,
                params,
                eval_metrics),
            config_metric
        )

        for metric in metrics:
            optim_params.pop(metric)

        m = model(**optim_params)
        m.fit(X_train, y_train)

        optim_metrics = {}
        optim_metrics = eval_metrics(
            m, X_validation, y_validation, optim_metrics)

        model_name = f"{m}".split("(")[0]  # fix this line

        save_model(
            m, folder + f"/{model_name}/", optim_params, optim_metrics)


maximise_metrics = ['R_sq', 'accuracy', 'precision', 'recall', 'f1']
minimise_metrics = ['RMSE']


def find_best_model(models, folder, metric):

    metrics_summary = []
    for model, config in models:
        model_name = f"{model}".split(".")[-1].strip("'>")  # fix this line
        filepath = folder + f'/{model_name}' + '/metrics.json'
        with open(filepath) as json_file:
            metrics = json.load(json_file)
            metrics_summary.append((model_name, metrics))

    opt_metric = metrics_summary[0][1][metric]
    opt_model_name = metrics_summary[0][0]

    for model_name, metrics in metrics_summary:

        if metric in maximise_metrics:
            if metrics[metric] > opt_metric:
                opt_metric = metrics[metric]
                opt_model_name = model_name

        elif metric in minimise_metrics:
            if metrics[metric] < opt_metric:
                opt_metric = metrics[metric]
                opt_model_name = model_name

    return opt_model_name, opt_metric


def load_model(folder, model_name):

    model_filepath = folder + f'/{model_name}'

    model = joblib.load(model_filepath + '/model.joblib')

    with open(model_filepath + '/hyperparameters.json') as json_file:
        params = json.load(json_file)

    with open(model_filepath + '/metrics.json') as json_file:
        metrics = json.load(json_file)

    return model, params, metrics
