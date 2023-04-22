import pandas as pd
import numpy as np
from tabular_data import load_airbnb, rating_columns, default_value_columns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import math
import csv
import itertools
import utils

import joblib
import os
import json

np.random.seed(2)

# as previously, move this into a generic file and reconfigure accordingly
df = pd.read_csv('clean_tabular_data.csv')
features, labels = load_airbnb(
    df, (rating_columns + default_value_columns), 'Price_Night')

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, train_size=0.7, test_size=0.3)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_test, y_test, test_size=0.5)

data = X_train, y_train, X_validation, y_validation, X_test, y_test

# your_model = LinearRegression()
# your_model.fit(x_train, y_train)
# print(your_model.coef_)
# print(your_model.intercept_)
# print(your_model.score(x_test, y_test))
epochs = 10000  # verifu how much difference varying number of epochs makes
# your_model = SGDRegressor() # max_iter=1000
# your_model.fit(X_train, y_train)
# print(your_model.coef_)
# print(your_model.intercept_)
# predicted_y_train = your_model.predict(X_train)
# predicted_y_test = your_model.predict(X_test)
# MSE_train = mean_squared_error(y_train, predicted_y_train)
# MSE_test = mean_squared_error(y_test, predicted_y_test)
# RMSE_train = math.sqrt(MSE_train)
# RMSE_test = math.sqrt(MSE_test)
# print(f"R^2 for the training data: {your_model.score(X_train, y_train)}")
# print(f"RMSE for the trainining data: {RMSE_train}")
# print(f"R^2 for the test data: {your_model.score(X_test, y_test)}")
# print(f"RMSE for the test data: {RMSE_test}")

# move into utils file and combine with classification if possible


def custom_tune_reg_model_hyperparams(
        model,
        data,
        hyperparams):

    X_train, y_train, X_validation, y_validation, X_test, y_test = data
    acc_metrics = []

    permutations = itertools.product(*list(hyperparams.values()))
    for permutation in permutations:
        acc = {key: value for key, value in zip(
            list(hyperparams.keys()), permutation)}
        reg_model = model(**acc)
        reg_model.fit(X_train, y_train)
        acc['R_sq'] = reg_model.score(X_validation, y_validation)
        acc['RMSE'] = math.sqrt(mean_squared_error(
            y_validation, reg_model.predict(X_validation)))
        acc_metrics.append(acc)

    return acc_metrics


def choose_optimal_hyperparams(metrics):
    return sorted(metrics, key=lambda m: m['RMSE'])[0]


# might be worth utilising this more
def tune_reg_model_hyperparams(
        model,
        data,
        hyperparams
):
    X_train, y_train, X_validation, y_validation, X_test, y_test = data
    reg_model = model()
    grid_reg = GridSearchCV(
        estimator=reg_model,
        param_grid=hyperparams,
        scoring='neg_root_mean_squared_error',
        cv=2)
    grid_reg.fit(X_train, y_train)

    print(" Results from Grid Search ")
    print("\n The best estimator across ALL searched params:\n",
          grid_reg.best_estimator_)
    print("\n The best score across ALL searched params:\n",
          grid_reg.best_score_, grid_reg.score(X_validation, y_validation))
    print("\n The best parameters across ALL searched params:\n",
          grid_reg.best_params_)

    return grid_reg

# to be moved to a generic utils file


def save_model(trained_model, opt_hyperparams, metrics, folder):
    filepath = folder + 'regression_model.joblib'
    joblib.dump(trained_model, filepath)

    filepath = folder + 'hyperparameters.json'
    json.dump(opt_hyperparams, open(filepath, 'w'))

    filepath = folder + 'metrics.json'
    json.dump(metrics, open(filepath, 'w'))

# with open('initial_results.csv', 'w') as output_file:
#     dict_writer = csv.DictWriter(output_file, param_lst[0].keys())
#     dict_writer.writeheader()
#     dict_writer.writerows(param_lst)


models = {
    SGDRegressor: utils.SGDRegressor_params,
    LinearRegression: utils.LinearRegression_params,
    DecisionTreeRegressor: utils.DecisionTreeRegressor_params,
    RandomForestRegressor: utils.RandomForestRegressor_params,
    GradientBoostingRegressor: utils.GradientBoostingRegressor_params
}

# models = {
#     GradientBoostingRegressor: utils.GradientBoostingRegressor_params
# }


def evaluate_all_models(models, data):

    X_train, y_train, X_validation, y_validation, X_test, y_test = data

    for model, params in models.items():
        optimal_params = choose_optimal_hyperparams(
            custom_tune_reg_model_hyperparams(model, data, params)
        )
        optimal_params.pop('R_sq')
        optimal_params.pop('RMSE')
        model = model(**optimal_params)
        model.fit(X_train, y_train)
        metrics = {
            'R_sq': model.score(X_validation, y_validation),
            'RMSE': math.sqrt(mean_squared_error(
                y_validation, model.predict(X_validation)))
        }
        model_name = f"{model}".split("(")[0]
        save_model(model, optimal_params, metrics,
                   f"models/regression/{model_name}/")


# evaluate_all_models(models, data)


def find_best_model(models, folder):
    opt_RMSE = 10000
    for model in models.keys():
        model_name = f"{model}".split(".")[-1].strip("'>")
        filepath = folder + model_name + '/metrics.json'
        with open(filepath) as json_file:
            metrics = json.load(json_file)
            print(model_name, metrics)
            if metrics['RMSE'] < opt_RMSE:
                opt_RMSE = metrics['RMSE']
                opt_model_name = model_name

    opt_model_filepath = folder + opt_model_name

    opt_model = joblib.load(opt_model_filepath + '/regression_model.joblib')

    with open(opt_model_filepath + '/hyperparameters.json') as json_file:
        opt_hyperparameters = json.load(json_file)

    with open(opt_model_filepath + '/metrics.json') as json_file:
        opt_metrics = json.load(json_file)

    return opt_model_name, opt_model, opt_hyperparameters, opt_metrics


opt_model_name, opt_model, opt_params, opt_metrics = find_best_model(
    models, f"models/regression/")
print(opt_model_name, opt_params, opt_metrics)

# fix this block
# if __name__ == "__main__":
#     print("hello")
