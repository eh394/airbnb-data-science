import pandas as pd
import numpy as np
from data_handling import load_df, load_split_X_y, rating_columns, default_value_columns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import math
import csv
import itertools
import config

import joblib
import os
import json

from utils import evaluate_all_models, regression_metrics, eval_metrics_regression, find_best_model, load_model

np.random.seed(2)

# Load Clean Data
df = load_df('listing.csv', 'clean_tabular_data.csv',
             rating_columns, 'Description', default_value_columns, 1)
X_train, y_train, X_validation, y_validation, X_test, y_test = load_split_X_y(
    df, (rating_columns + default_value_columns), 'Price_Night', 0.7, 0.5)

epochs = 1000

# Define models to consider
models = [
    (SGDRegressor, config.SGDRegressor_params),
    (LinearRegression, config.LinearRegression_params),
    (DecisionTreeRegressor, config.DecisionTreeRegressor_params),
    (RandomForestRegressor, config.RandomForestRegressor_params),
    (GradientBoostingRegressor, config.GradientBoostingRegressor_params)
]


if __name__ == "__main__":
    
    evaluate_all_models(models, X_train, y_train, X_validation, y_validation, regression_metrics, eval_metrics_regression, 'models/regression', 'RMSE')

    opt_model_name, opt_metric = find_best_model(
        models, 'models/regression', 'RMSE')

    opt_model, opt_params, opt_metrics = load_model(
        'models/regression', opt_model_name)
    
    print(opt_model, opt_params, opt_metrics)










#
# def tune_reg_model_hyperparams(
# model,
# data,
# hyperparams
# ):
# X_train, y_train, X_validation, y_validation, X_test, y_test = data
# reg_model = model()
# grid_reg = GridSearchCV(
# estimator=reg_model,
# param_grid=hyperparams,
# scoring='neg_root_mean_squared_error',
# cv=2)
# grid_reg.fit(X_train, y_train)
#
# print(" Results from Grid Search ")
# print("\n The best estimator across ALL searched params:\n",
#   grid_reg.best_estimator_)
# print("\n The best score across ALL searched params:\n",
#   grid_reg.best_score_, grid_reg.score(X_validation, y_validation))
# print("\n The best parameters across ALL searched params:\n",
#   grid_reg.best_params_)
#
# return grid_reg
