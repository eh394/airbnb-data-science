import pandas as pd
import numpy as np
from tabular_data import load_airbnb, rating_columns, default_value_columns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import math
import csv
import itertools
from utils import hyperparams
import joblib
import os
import json

np.random.seed(2)

df = pd.read_csv('clean_tabular_data.csv')
features, labels = load_airbnb(
    df, (rating_columns + default_value_columns), 'Price_Night')

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, train_size=0.7, test_size=0.3)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_test, y_test, test_size=0.5)

# your_model = LinearRegression()
# your_model.fit(x_train, y_train)
# print(your_model.coef_)
# print(your_model.intercept_)
# print(your_model.score(x_test, y_test))
epochs = 10000
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


# def custom_tune_regression_model_hyperparameters(
#         model_class,
#         data,
#         hyperparams_dict):

#     data = X_train, y_train, X_validation, y_validation, X_test, y_test
#     hyperparams_metrics_lst = []

#     permutations = itertools.product(*list(hyperparams_dict.values()))
#     for permutation in permutations:
#         permutation_dict = {key: value for key, value in zip(
#             list(hyperparams_dict.keys()), permutation)}
#         reg_model = model_class(**permutation_dict)
#         reg_model.fit(X_train, y_train)
#         permutation_dict['R_sq'] = reg_model.score(X_validation, y_validation)
#         permutation_dict['RMSE'] = math.sqrt(mean_squared_error(
#             y_validation, reg_model.predict(X_validation)))
#         hyperparams_metrics_lst.append(permutation_dict)

#     return hyperparams_metrics_lst


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

# data = X_train, y_train, X_validation, y_validation, X_test, y_test
# hyperparams_metrics_lst = custom_tune_reg_model_hyperparams(
#     SGDRegressor, data, hyperparams)


# def choose_optimal_hyperparams(metrics):
#     optimal_RMSE = metrics[0]['RMSE']
#     optimal_hyperparams = metrics[0]
#     for metric in metrics:
#         if metric['RMSE'] < optimal_RMSE:
#             optimal_RMSE = metric['RMSE']
#             optimal_hyperparams = metric
#     return optimal_hyperparams

def choose_optimal_hyperparams(metrics):
    return sorted(metrics, key=lambda m: m['RMSE'])[0]


# optimal_model_hyperparams = choose_optimal_hyperparams(
#     hyperparams_metrics_lst)
# print(optimal_model_hyperparams)


def tune_reg_model_hyperparams(
        model,
        data,
        hyperparams
):
    X_train, y_train, X_validation, y_validation, X_test, y_test = data
    reg_model = model()
    grid_reg = GridSearchCV(estimator=reg_model, param_grid=hyperparams, cv=2)
    grid_reg.fit(X_train, y_train)
   
    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",grid_reg.best_estimator_)
    print("\n The best score across ALL searched params:\n",grid_reg.best_score_)
    print("\n The best parameters across ALL searched params:\n",grid_reg.best_params_)

    return grid_reg


best_reg=SGDRegressor(
    loss='squared_epsilon_insensitive',
    penalty='l2',
    learning_rate='invscaling',
    alpha=0.01,
    max_iter=1000,
    early_stopping=True)

best_reg.fit(X_train, y_train)

optimal_model_hyperparams = {
    'loss': 'squared_epsilon_insensitive',
    'penalty': 'l2',
    'learning_rate': 'invscaling',
    'alpha': 0.01,
    'max_iter': 1000,
    'early_stopping': True
}

optimal_metrics = {
     'R_sq': best_reg.score(X_validation, y_validation),
     'RMSE': math.sqrt(mean_squared_error(
            y_validation, best_reg.predict(X_validation)))
}


# def save_model(folder, trained_model, filename):
#     dirname = os.path.dirname(filename)
#     filename = os.path.join(dirname, folder)
#     joblib.dump(trained_model, filename)
    

# save_model('./models/regression/linear_regression', best_reg, 'regression_model.joblib')

def save_model(trained_model, opt_hyperparams, metrics, folder):
    filepath = folder + 'regression_model.joblib'
    joblib.dump(trained_model, filepath)

    filepath = folder + 'hyperparameters.json'
    json.dump(opt_hyperparams, open(filepath, 'w'))

    filepath = folder + 'metrics.json'
    json.dump(metrics, open(filepath, 'w'))
    

save_model(best_reg, optimal_model_hyperparams, optimal_metrics, 'models/regression/linear_regression/')


# with open('initial_results.csv', 'w') as output_file:
#     dict_writer = csv.DictWriter(output_file, param_lst[0].keys())
#     dict_writer.writeheader()
#     dict_writer.writerows(param_lst)

SGDRegressor
LinearRegression
DecisionTreeRegressor
RandomForestRegressor
GradientBoostingRegressor
