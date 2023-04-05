import pandas as pd
import numpy as np
from tabular_data import load_airbnb, rating_columns, default_value_columns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import math
import csv
import itertools
from utils import hyperparams_dict

np.random.seed(2)

df = pd.read_csv('clean_tabular_data.csv')
features, labels = load_airbnb(df, (rating_columns + default_value_columns), 'Price_Night') 

X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, test_size=0.3)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5)

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





# print(len(X_train))
# print(len(X_validation))
# print(len(X_test))


# hyper_lst = list(hyperparams_dict.keys())
# hyper_params = list(hyperparams_dict.values())

# print(hyper_lst)
# print(hyper_params)

param_lst = []

permutations = itertools.product(*list(hyperparams_dict.values()))
for permutation in permutations:
    foo_dict = {key: value for key, value in zip(list(hyperparams_dict.keys()), permutation)}
    sgd_lr = SGDRegressor(**foo_dict)
    sgd_lr.fit(X_train, y_train)
    foo_dict['R_sq'] = sgd_lr.score(X_validation, y_validation)
    foo_dict['RMSE'] = math.sqrt(mean_squared_error(y_validation, sgd_lr.predict(X_validation)))
    param_lst.append(foo_dict)


# loss = ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
# penalty = ['l2', 'l1', 'elasticnet', None]
# learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
# alpha = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
# max_iter = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
# early_stopping = [True, False]


# def custom_tune_regression_model_hyperparameters(
#         model_class,
#         X_train,
#         y_train
#         X_validation,
#         y_validation,
#         X_test,
#         y_test
#         hyperparameters_dictionary):
#     pass

# param_lst = []
def make_sgdreg(**kwargs):
    SGDRegressor(**kwargs)
# for loss_param, penalty_param, rate_param, alpha_param, iter_param, stop_param in itertools.product(*hyper_params):
#     sgd_lr = SGDRegressor(
#           loss=loss_param,
#           penalty=penalty_param,
#           learning_rate=rate_param,
#           alpha=alpha_param,
#           max_iter=iter_param,
#           early_stopping=stop_param)
#     sgd_lr.fit(X_train, y_train)
#     param_lst.append({
#         'loss': loss_param,
#         'penalty': penalty_param, 
#         'learning_rate' : rate_param,
#         'alpha': alpha_param,
#         'max_iter': iter_param,
#         'early_stopping': stop_param,
#         'R^2': sgd_lr.score(X_validation, y_validation),
#         'RMSE': math.sqrt(mean_squared_error(y_validation, sgd_lr.predict(X_validation)))
#         })
    


# for loss_param in loss:
#     for penalty_param in penalty:
#         for rate_param in learning_rate:
#             for alpha_param in alpha:
#                 for iter_param in max_iter:
#                     for stop_param in early_stopping:
#                         sgd_lr = SGDRegressor(
#                             loss=loss_param,
#                             penalty=penalty_param,
#                             learning_rate=rate_param)
#                         sgd_lr.fit(X_train, y_train)
#                         param_lst.append({
#                             'loss': loss_param,
#                             'penalty': penalty_param,
#                             'learning_rate' : rate_param,
#                             'alpha': alpha_param,
#                             'max_iter': iter_param,
#                             'early_stopping': stop_param,
#                             'R^2': sgd_lr.score(X_validation, y_validation),
#                             'RMSE': math.sqrt(mean_squared_error(y_validation, sgd_lr.predict(X_validation)))
#                         })


# print(param_lst)

# with open('initial_results.csv', 'w') as output_file:
#     dict_writer = csv.DictWriter(output_file, param_lst[0].keys())
#     dict_writer.writeheader()
#     dict_writer.writerows(param_lst)


