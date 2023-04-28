import pandas as pd
import numpy as np
from data_handling import load_df, load_split_X_y, rating_columns, default_value_columns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import math
import itertools
import joblib
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


from sklearn.linear_model import LogisticRegression
import config

from utils import evaluate_all_models, classification_metrics, eval_metrics_classification, find_best_model, load_model


np.random.seed(2)

# Load Clean Data
df = load_df('listing.csv', 'clean_tabular_data.csv',
             rating_columns, 'Description', default_value_columns, 1)
X_train, y_train, X_validation, y_validation, X_test, y_test = load_split_X_y(
    df, (rating_columns + default_value_columns), 'Price_Night', 0.7, 0.5)



epochs = 10000

# Define models to consider
models = [
    (LogisticRegression, config.LogisticRegression_params),
    (DecisionTreeClassifier, config.DecisionTreeClassifier_params),
    (RandomForestClassifier, config.RandomForestClassifier_params),
    (GradientBoostingClassifier, config.GradientBoostingClassifier_params)
]


if __name__ == "__main__":
    
    evaluate_all_models(models, X_train, y_train, X_validation, y_validation, classification_metrics, eval_metrics_classification, 'models/classification', 'accuracy')

    opt_model_name, opt_metric = find_best_model(
        models, 'models/classification', 'accuracy')

    opt_model, opt_params, opt_metrics = load_model(
        'models/classification', opt_model_name)
    
    print(opt_model, opt_params, opt_metrics)


# Incorporate regularization, i.e. you want to ensure that the metrics of the training set and the validation set are of similar value to prevent overfitting
# Also incorporate last step - reporting to provide metrics on the test set

# CONSIDER HOW TO HANDLE INCOMPATIBLE HYPERPARAMETERS


# def evaluate_all_models(
#         models,
#     X_train,
#     y_train,
#     X_validation,
#     y_validation
# ):

#     for model, params in models.items():
#         optim_params = choose_optimal_hyperparams(
#             custom_tune_model_params(model, data, params) # fix this line, function changed
#         )
#         optim_params.pop('accuracy')
#         optim_params.pop('precision')
#         optim_params.pop('recall')
#         optim_params.pop('f1')
#         model = model(**optim_params)
#         model.fit(X_train, y_train)
#         metrics = {
#             'accuracy': accuracy_score(y_validation, model.predict(X_validation)),
#             'precision': precision_score(y_validation, model.predict(X_validation), average='weighted')
#         }
#         model_name = f"{model}".split("(")[0]
#         save_model(model, optimal_params, metrics,
#                    f"models/classification/{model_name}/")


# # evaluate_all_models(models, data)


# def find_best_model(models, folder):
#     opt_accuracy = 0
#     for model in models.keys():
#         model_name = f"{model}".split(".")[-1].strip("'>")
#         filepath = folder + model_name + '/metrics.json'
#         with open(filepath) as json_file:
#             metrics = json.load(json_file)
#             print(model_name, metrics)
#             if metrics['accuracy'] > opt_accuracy:
#                 opt_accuracy = metrics['accuracy']
#                 opt_model_name = model_name

#     opt_model_filepath = folder + opt_model_name

#     opt_model = joblib.load(opt_model_filepath +
#                             '/classification_model.joblib')

#     with open(opt_model_filepath + '/hyperparameters.json') as json_file:
#         opt_hyperparameters = json.load(json_file)

#     with open(opt_model_filepath + '/metrics.json') as json_file:
#         opt_metrics = json.load(json_file)

#     return opt_model_name, opt_model, opt_hyperparameters, opt_metrics


# opt_model_name, opt_model, opt_params, opt_metrics = find_best_model(
#     models, f"models/classification/")
# print(opt_model_name, opt_params, opt_metrics)


# # foo = custom_tune_reg_model_hyperparams(data, config.LogisticRegression_params)
# # foo = choose_optimal_hyperparams(custom_tune_reg_model_hyperparams(LogisticRegression, data, config.LogisticRegression_params))
# # print(foo)
