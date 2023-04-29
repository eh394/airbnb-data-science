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
    df, (rating_columns + default_value_columns), 'Category', 0.7, 0.5)


epochs = 1000

# Define models to consider
models = [
    (LogisticRegression, config.LogisticRegression_params),
    (DecisionTreeClassifier, config.DecisionTreeClassifier_params),
    (RandomForestClassifier, config.RandomForestClassifier_params),
    (GradientBoostingClassifier, config.GradientBoostingClassifier_params)
]


if __name__ == "__main__":

    evaluate_all_models(models, X_train, y_train, X_validation, y_validation,
                        classification_metrics, eval_metrics_classification, 'models/classification', 'accuracy')

    opt_model_name, opt_metric = find_best_model(
        models, 'models/classification', 'accuracy')

    opt_model, opt_params, opt_metrics = load_model(
        'models/classification', opt_model_name)

    print(opt_model, opt_params, opt_metrics)


# Incorporate regularization, i.e. you want to ensure that the metrics of the training set and the validation set are of similar value to prevent overfitting
# Also incorporate last step - reporting to provide metrics on the test set

# CONSIDER HOW TO HANDLE INCOMPATIBLE HYPERPARAMETERS
