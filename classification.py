import pandas as pd
import numpy as np
from tabular_data import load_airbnb, rating_columns, default_value_columns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import math
import itertools
import joblib
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


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

epochs = 10000
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
    return sorted(metrics, key=lambda m: m['accuracy'], reverse=True)[0]



def save_model(trained_model, opt_hyperparams, metrics, folder):
    filepath = folder + 'classification_model.joblib'
    joblib.dump(trained_model, filepath)

    filepath = folder + 'hyperparameters.json'
    json.dump(opt_hyperparams, open(filepath, 'w'))

    filepath = folder + 'metrics.json'
    json.dump(metrics, open(filepath, 'w'))


models = {
    LogisticRegression: utils.LogisticRegression_params,
    DecisionTreeClassifier: utils.DecisionTreeClassifier_params,
    RandomForestClassifier: utils.RandomForestClassifier_params,
    GradientBoostingClassifier: utils.GradientBoostingClassifier_params
}

# Incorporate regularization, i.e. you want to ensure that the metrics of the training set and the validation set are of similar value to prevent overfitting
# Also incorporate last step - reporting to provide metrics on the test set

# CONSIDER HOW TO HANDLE INCOMPATIBLE HYPERPARAMETERS

def evaluate_all_models(models, data):

    X_train, y_train, X_validation, y_validation, X_test, y_test = data

    for model, params in models.items():
        optimal_params = choose_optimal_hyperparams(
            custom_tune_reg_model_hyperparams(model, data, params)
        )
        optimal_params.pop('accuracy')
        optimal_params.pop('precision')
        optimal_params.pop('recall')
        optimal_params.pop('f1')
        model = model(**optimal_params)
        model.fit(X_train, y_train)
        metrics = {
            'accuracy': accuracy_score(y_validation, model.predict(X_validation)),
            'precision': precision_score(y_validation, model.predict(X_validation), average='weighted')
        }
        model_name = f"{model}".split("(")[0]
        save_model(model, optimal_params, metrics,
                   f"models/classification/{model_name}/")


# evaluate_all_models(models, data)


def find_best_model(models, folder):
    opt_accuracy = 0
    for model in models.keys():
        model_name = f"{model}".split(".")[-1].strip("'>")
        filepath = folder + model_name + '/metrics.json'
        with open(filepath) as json_file:
            metrics = json.load(json_file)
            print(model_name, metrics)
            if metrics['accuracy'] > opt_accuracy:
                opt_accuracy = metrics['accuracy']
                opt_model_name = model_name

    opt_model_filepath = folder + opt_model_name

    opt_model = joblib.load(opt_model_filepath + '/classification_model.joblib')

    with open(opt_model_filepath + '/hyperparameters.json') as json_file:
        opt_hyperparameters = json.load(json_file)

    with open(opt_model_filepath + '/metrics.json') as json_file:
        opt_metrics = json.load(json_file)

    return opt_model_name, opt_model, opt_hyperparameters, opt_metrics


opt_model_name, opt_model, opt_params, opt_metrics = find_best_model(models, f"models/classification/")
print(opt_model_name, opt_params, opt_metrics)


# foo = custom_tune_reg_model_hyperparams(data, utils.LogisticRegression_params)
# foo = choose_optimal_hyperparams(custom_tune_reg_model_hyperparams(LogisticRegression, data, utils.LogisticRegression_params))
# print(foo)


