import datetime
import itertools
import joblib
import json
import math
import os
import torch
from sklearn.metrics import mean_squared_error, precision_score, recall_score, accuracy_score, f1_score


def custom_tune_model_params(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        params,
        derive_metrics
):

    acc_metrics = []

    configs = itertools.product(*list(params.values()))
    for config in configs:
        acc = {key: value for key, value in zip(
            list(params.keys()), config)}

        m = model(**acc)
        m.fit(X_train, y_train)

        acc = acc.update(derive_metrics(m, X_validation, y_validation))
        acc_metrics.append(acc)

    return acc_metrics


def derive_metrics_regression(model, X_validation, y_validation):
    metrics = {}
    metrics["R_sq"] = model.score(X_validation, y_validation)
    metrics["RMSE"] = math.sqrt(mean_squared_error(
        y_validation, model.predict(X_validation)))
    return metrics


def derive_metrics_classification(model, X_validation, y_validation):
    metrics = {}
    metrics["accuracy"] = accuracy_score(
        y_validation, model.predict(X_validation))
    metrics["precision"] = precision_score(
        y_validation, model.predict(X_validation), average="weighted")
    metrics["recall"] = recall_score(
        y_validation, model.predict(X_validation), average="weighted")
    metrics["f1"] = f1_score(y_validation, model.predict(
        X_validation), average="weighted")
    return metrics


def choose_optimal_params(acc_metrics, config_metric):

    assert (config_metric in regression_metrics) or (
        config_metric in classification_metrics)
    if config_metric in minimise_metrics:
        optim_params = sorted(
            acc_metrics, key=lambda m: m[config_metric], reverse=False)[0]
    elif config_metric in maximise_metrics:
        optim_params = sorted(
            acc_metrics, key=lambda m: m[config_metric], reverse=True)[0]
    return optim_params


def save_model(trained_model, output_folder, opt_hyperparams=None, metrics=None):

    # if isinstance(trained_model, NN) == True:
    if False == True:
        now = datetime.datetime.now()
        output_folder = f"{output_folder}/{now}"
        os.makedirs(output_folder)
        filepath = f"{output_folder}/model.pt"
        torch.save(trained_model.state_dict(), filepath)

    else:
        filepath = f"{output_folder}/model.joblib"
        joblib.dump(trained_model, filepath)

    filepath = f"{output_folder}/hyperparameters.json"
    json.dump(opt_hyperparams, open(filepath, "w"))

    filepath = f"{output_folder}/metrics.json"
    json.dump(metrics, open(filepath, "w"))


def evaluate_all_models(
    models,
    X_train,
    y_train,
    X_validation,
    y_validation,
    metrics,
    derive_metrics,
    output_folder,
    config_metric
):

    for model, params in models:

        optim_params = choose_optimal_params(
            custom_tune_model_params(  # consider assigning it to a variable as bumhead said
                model,
                X_train,
                y_train,
                X_validation,
                y_validation,
                params,
                derive_metrics),
            config_metric
        )

        for metric in metrics:
            optim_params.pop(metric)

        m = model(**optim_params)
        m.fit(X_train, y_train)

        optim_metrics = {}
        optim_metrics = derive_metrics(
            m, X_validation, y_validation, optim_metrics)

        model_name = f"{m}".split("(")[0]  # how to better approach this

        save_model(
            m, f"{output_folder}/{model_name}/", optim_params, optim_metrics)


def find_optimal_model(models, output_folder, metric):

    metrics_summary = []
    for model, config in models:
        # how to better approach this?
        model_name = f"{model}".split(".")[-1].strip("'>")
        filepath = f"{output_folder}/{model_name}/metrics.json"
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

    model_filepath = f"{folder}/{model_name}"

    model = joblib.load(f"{model_filepath}/model.joblib")

    with open(f"{model_filepath}/hyperparameters.json") as json_file:
        params = json.load(json_file)

    with open(f"{model_filepath}/metrics.json") as json_file:
        metrics = json.load(json_file)

    return model, params, metrics
