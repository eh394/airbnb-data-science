import pandas as pd
import numpy as np
from tabular_data import load_airbnb, rating_columns, default_value_columns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import math
import itertools

from sklearn.linear_model import LogisticRegression
import utils

np.random.seed(2)


df = pd.read_csv('clean_tabular_data.csv')
features, labels = load_airbnb(
    df, (rating_columns + default_value_columns), 'Category')


X_train, X_test, y_train, y_test = train_test_split(
    features, labels, train_size=0.7, test_size=0.3)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_test, y_test, test_size=0.5)

data = X_train, y_train, X_validation, y_validation, X_test, y_test

# epochs = 10000
# model = LogisticRegression(max_iter=2000)
# model.fit(X_train, y_train)
# print(model.coef_)
# print(model.intercept_)
# print(model.score(X_test, y_test))


# predicted_y_train = model.predict(X_train)
# predicted_y_test = model.predict(X_test)

# accuracy_y_train = accuracy_score(y_train, predicted_y_train)
# accuracy_y_test = accuracy_score(y_test, predicted_y_test)
# precision_y_train = precision_score(y_train, predicted_y_train, average='weighted')
# precision_y_test = precision_score(y_test, predicted_y_test, average='weighted')
# recall_y_train = recall_score(y_train, predicted_y_train, average='weighted')
# recall_y_test = recall_score(y_test, predicted_y_test, average='weighted')
# f1_y_train = f1_score(y_train, predicted_y_train, average='weighted')
# f1_y_test = f1_score(y_test, predicted_y_test, average='weighted')


# print(f"R^2 for the training data: {model.score(X_train, y_train)}")
# print(f"accuracy of the training data: {accuracy_y_train} /n precision of the training data: {precision_y_train} /n recall of the training data: {recall_y_train} /n f1 of the training data: {f1_y_train}")

# print(f"R^2 for the test data: {model.score(X_test, y_test)}")
# print(f"accuracy of the test data: {accuracy_y_test} /n precision of the test data: {precision_y_test} /n recall of the test data: {recall_y_test} /n f1 of the test data: {f1_y_test}")



# def custom_tune_reg_model_hyperparams(
#         model,
#         data,
#         hyperparams):

#     X_train, y_train, X_validation, y_validation, X_test, y_test = data
#     acc_metrics = []

#     permutations = itertools.product(*list(hyperparams.values()))
#     for permutation in permutations:
#         acc = {key: value for key, value in zip(
#             list(hyperparams.keys()), permutation)}
#         model = model(**acc)
#         model.fit(X_train, y_train)
#         acc['accuracy'] = accuracy_score(y_validation, model.predict(X_validation))
#         acc['precision'] = precision_score(y_validation, model.predict(X_validation), average='weighted')
#         acc['recall'] = recall_score(y_validation, model.predict(X_validation), average='weighted')
#         acc['f1'] = f1_score(y_validation, model.predict(X_validation), average='weighted')
#         acc_metrics.append(acc)

#     return acc_metrics


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
        acc['accuracy'] = accuracy_score(y_validation, reg_model.predict(X_validation))
        acc['precision'] = precision_score(y_validation, reg_model.predict(X_validation), average='weighted')
        acc['recall'] = recall_score(y_validation, reg_model.predict(X_validation), average='weighted')
        acc['f1'] = f1_score(y_validation, reg_model.predict(X_validation), average='weighted')
        acc_metrics.append(acc)

    return acc_metrics


def choose_optimal_hyperparams(metrics):
    return sorted(metrics, reverse=True, key=lambda m: m['accuracy'])[0]


# foo = custom_tune_reg_model_hyperparams(data, utils.LogisticRegression_params)
foo = choose_optimal_hyperparams(custom_tune_reg_model_hyperparams(LogisticRegression, data, utils.LogisticRegression_params))
print(foo)


