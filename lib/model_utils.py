import datetime
import itertools
import joblib
import json
import math
import os
from sklearn.metrics import mean_squared_error, precision_score, recall_score, accuracy_score, f1_score

from config import model_config


def custom_tune_model_params(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        params,
        derive_metrics
):
    """Evaluates performance metrics for various permutations of hyperparameters for a given model class.

    Parameters
    ----------
    model : class sklearn.(linear_model | tree | ensemble)
        model class to tune hyperparameters for, for instance SGDRegressor
    X_train : pd.DataFrame
        feature data used to train models
    y_train : pd.Series
        label data used to train models
    X_validation : pd.DataFrame
        feature data used to evaluate models
    y_validation : pd.Series
        label data used to evaluate models
    params : dict<str, list>
        key: hyperparameter name
        value: hyperparameter options to evaluate
    derive_metrics : fn (derive_metrics_regression | derive_metrics_classification)

    Returns
    -------
    acc_metrics: list<dict>
        List of hyperparameter permutations and their corresponding metrics. 
    """
    acc_metrics = []
    configs = itertools.product(*list(params.values()))
    for config in configs:
        acc = {key: value for key, value in zip(
            list(params.keys()), config)}
        m = model(**acc)
        m.fit(X_train, y_train)
        acc.update(derive_metrics(m, X_validation, y_validation))
        acc_metrics.append(acc)
    return acc_metrics


def derive_metrics_regression(model, X_validation, y_validation):
    """Derives performance metrics for a regression model.

    Parameters
        ----------
        model : class instance sklearn.(linear_model | tree | ensemble)
            a trained sklearn model
        X_validation : pd.DataFrame
            feature data used to evaluate models
        y_validation : pd.Series
            label data used to evaluate models

    Returns
    -------
    metrics: dict<str, float>
        key: metric name
        value: metric value
    """

    metrics = {}
    metrics["R_sq"] = model.score(X_validation, y_validation)
    metrics["RMSE"] = math.sqrt(mean_squared_error(
        y_validation, model.predict(X_validation)))
    return metrics


def derive_metrics_classification(model, X_validation, y_validation):
    """Derives performance metrics for a classification model.
    
    Parameters
        ----------
        model : class instance sklearn.(linear_model | tree | ensemble)
            a trained sklearn model
        X_validation : pd.DataFrame
            feature data used to evaluate models
        y_validation : pd.Series
            label data used to evaluate models

    Returns
    -------
    metrics: dict<str, float>
        key: metric name
        value: metric value
    """

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
    """Using a metric name, selects optimum hyperparameters from hyperparameter permutations.

    Parameters
    ----------
    acc_metrics: list<dict>
        List of hyperparameter permutations and their corresponding metrics. 
    config_metric : str
        metric name

    Returns
    -------
    optim_params : dict
        A dictionary of hyperparameters and  metrics corresponding to a model with the optimal chosen metric.    
    """
    assert (config_metric in model_config.regression_metrics) or (
        config_metric in model_config.classification_metrics)
    if config_metric in model_config.minimise_metrics:
        optim_params = sorted(
            acc_metrics, key=lambda m: m[config_metric], reverse=False)[0]
    elif config_metric in model_config.maximise_metrics:
        optim_params = sorted(
            acc_metrics, key=lambda m: m[config_metric], reverse=True)[0]
    return optim_params


def save_model(trained_model, output_folder, opt_hyperparams=None, metrics=None):
    """Saves a trained model, its hyperparameters and performance metrics to a directory.
    
    Parameters
    ----------
    trained_model : class instance sklearn.(linear_model | tree | ensemble)
        a trained sklearn model
    output_folder : str
        filepath to output folder
    opt_hyperparameters : dict, optional
        a dictionary of model hyperparameters (default is None)
    metrics : dict, optional
        a dictionary of model metrics (default is None)
    """
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
    """For each model class, the function tunes hyperparameters. 
    Using a metric names selects and saves optimal models.  
    
    Parameters
    ----------
    models : list<(class sklearn.(linear_model | tree | ensemble), dict)>
        key: hyperparameter name
        value: hyperparameter options to evaluate
    X_train : pd.DataFrame
        feature data used to train models
    y_train : pd.Series
        label data used to train models
    X_validation : pd.DataFrame
        feature data used to evaluate models
    y_validation : pd.Series
        label data used to evaluate models
    metrics : model_config.(regression_metrics | classification_metrics)
    derive_metrics : fn (derive_metrics_regression | derive_metrics_classification)
    output_folder : str
        filepath to output folder
    config_metric : str
        metric name
    """

    for model, params in models:
        metrics_all_models = custom_tune_model_params(
            model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            params,
            derive_metrics
        )

        optim_params = choose_optimal_params(
            metrics_all_models,
            config_metric
        )

        for metric in metrics:
            optim_params.pop(metric)
        m = model(**optim_params)
        m.fit(X_train, y_train)
        optim_metrics = {}
        optim_metrics = derive_metrics(
            m, X_validation, y_validation)
        model_name = f"{m}".split("(")[0]  # how to better approach this?

        save_model(
            m, f"{output_folder}/{model_name}/", optim_params, optim_metrics)


def find_optimal_model(models, output_folder, config_metric):
    """Using a metric name, finds optimal model from saved models.


    Parameters
    ----------
    models : list<(class sklearn.(linear_model | tree | ensemble), dict)>
        key: hyperparameter name
        value: hyperparameter options to evaluate
    output_folder : str
        filepath to output folder
    config_metric : str
        metric name

    Returns
    -------
    opt_model_name : str
        name of the optimal model class
    opt_metric : float
        value of the optimal (chosen) metric
    """
    
    metrics_summary = []
    for model, config_ in models:
        # how to better approach this?
        model_name = f"{model}".split(".")[-1].strip("'>")
        filepath = f"{output_folder}/{model_name}/metrics.json"
        with open(filepath) as json_file:
            metrics = json.load(json_file)
            metrics_summary.append((model_name, metrics))
    opt_metric = metrics_summary[0][1][config_metric]
    opt_model_name = metrics_summary[0][0]

    for model_name, metrics in metrics_summary:
        if config_metric in model_config.maximise_metrics:
            if metrics[config_metric] > opt_metric:
                opt_metric = metrics[config_metric]
                opt_model_name = model_name
        elif config_metric in model_config.minimise_metrics:
            if metrics[config_metric] < opt_metric:
                opt_metric = metrics[config_metric]
                opt_model_name = model_name
    return opt_model_name, opt_metric


def load_model(model_name, folder):
    model_filepath = f"{folder}/{model_name}"
    model = joblib.load(f"{model_filepath}/model.joblib")
    with open(f"{model_filepath}/hyperparameters.json") as json_file:
        params = json.load(json_file)
    with open(f"{model_filepath}/metrics.json") as json_file:
        metrics = json.load(json_file)
    return model, params, metrics
