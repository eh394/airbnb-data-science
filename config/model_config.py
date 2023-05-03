regression_metrics = ['R_sq', 'RMSE']
classification_metrics = ['accuracy', 'precision', 'recall', 'f1']

maximise_metrics = ['R_sq', 'accuracy', 'precision', 'recall', 'f1']
minimise_metrics = ['RMSE']


SGDRegressor_params = {
    'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet', None],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    'max_iter': [2000, 2500, 3000, 3500, 4000, 4500, 5000],
    'early_stopping': [True, False]
}


LinearRegression_params = {
    'fit_intercept': [True],
    'copy_X': [True],
    'n_jobs': [None],
    'positive': [False]
}


DecisionTreeRegressor_params = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 1, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 0.05, 0.10, 0.15, 0.20],
    'min_samples_leaf': [1, 0.05, 0.10, 0.15, 0.20],
    'max_features': [None, 'sqrt', 'log2'],
    'ccp_alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
}


RandomForestRegressor_params = {
    'n_estimators': [5, 10, 50, 100],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'max_depth': [None, 1, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 0.05, 0.10, 0.15, 0.20],
    'min_samples_leaf': [1, 0.05, 0.10, 0.15, 0.20],
    'max_features': [0.25, 0.5, 1.0, 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'ccp_alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
}


GradientBoostingRegressor_params = {
    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    'learning_rate': [0.05, 0.1, 1],
    'n_estimators': [50, 100, 500],
    'criterion': ['squared_error', 'friedman_mse'],
    'min_samples_split': [2, 0.10, 0.20],
    'min_samples_leaf': [1, 0.10, 0.20],
    'max_depth': [None, 1, 2, 3, 4, 5, 6],
    'max_features': [0.25, 0.5, 0.9, 'sqrt', 'log2', None]
}


LogisticRegression_params = {
    'penalty': ['l1', 'l2', None],
    'class_weight': ['balanced', None],
    'solver': ['saga'],
    'max_iter': [5000, 10000],
    'multi_class': ['ovr', 'multinomial']
}

DecisionTreeClassifier_params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 1, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 0.10, 0.20],
    'min_samples_leaf': [1, 0.10, 0.20],
    'max_features': [0.25, 0.5, 0.9, 'sqrt', 'log2', None],
    'ccp_alpha': [0.0, 0.01, 0.1]  # appears to be costly parameter
}

RandomForestClassifier_params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    # with this undefined, but criterion full set and class_weight as balanced, the result is better
    'max_depth': [None, 1, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 0.10, 0.20],
    'min_samples_leaf': [1, 0.10, 0.20],
    'max_features': [0.25, 0.5, 0.9, 'sqrt', 'log2', None],
    'class_weight': [None, 'balanced'],
    'ccp_alpha': [0.0, 0.01, 0.1]
}


GradientBoostingClassifier_params = {
    'loss': ['log_loss'],
    'learning_rate': [0.1, 1, 100, 10000],
    'n_estimators': [1, 100, 1000, 10000],
    'criterion': ['friedman_mse', 'squared_error'],
    'min_samples_split': [2, 0.10, 0.20],
    'min_samples_leaf': [1, 0.10, 0.20],
    'max_depth': [None, 1, 2, 3, 4, 5, 6],
    'max_features': [0.25, 0.5, 0.9, 'sqrt', 'log2', None],
    'ccp_alpha': [0.0, 0.01, 0.1]
}


NN_params = {
    'optimiser': ['SGD', 'Adam'],
    'lr': [0.0001, 0.00001, 0.000001],
    'hidden_layer_width': [40, 80],
    'model_depth': [3, 5, 7],
    'epochs': [10, 1000, 5000],
    'batch_size': [10, 20]
}


# REDUCED PARAMETERS FOR TEST RUNS

# SGDRegressor_params = {
#     'loss': ['squared_error'],
#     'penalty': ['l2'],
#     'learning_rate': ['optimal'],
#     'alpha': [0.01],
#     'max_iter': [5000],
#     'early_stopping': [False]
# }


# LinearRegression_params = {
#     'fit_intercept': [True],
#     'copy_X': [True],
#     'n_jobs': [None],
#     'positive': [False]
# }


# DecisionTreeRegressor_params = {
#     'criterion': ['squared_error'],
#     'splitter': ['best'],
#     'max_depth': [3],
#     'min_samples_split': [0.20],
#     'min_samples_leaf': [0.20],
#     'max_features': ['sqrt'],
#     'ccp_alpha': [0.01]
# }


# RandomForestRegressor_params = {
#     'n_estimators': [100],
#     'criterion': ['squared_error'],
#     'max_depth': [4],
#     'min_samples_split': [0.20],
#     'min_samples_leaf': [0.20],
#     'max_features': ['sqrt'],
#     'bootstrap': [False],
#     'ccp_alpha': [0.01]
# }


# GradientBoostingRegressor_params = {
#     'loss': ['squared_error'],
#     'learning_rate': [0.1],
#     'n_estimators': [100],
#     'criterion': ['squared_error'],
#     'min_samples_split': [0.20],
#     'min_samples_leaf': [0.20],
#     'max_depth': [4],
#     'max_features': ['sqrt']
# }

# LogisticRegression_params = {
#     'penalty': ['l2'],
#     'class_weight': ['balanced'],
#     'solver': ['saga'],
#     'max_iter': [5000],
#     'multi_class': ['multinomial']
# }

# DecisionTreeClassifier_params = {
#     'criterion': ['gini'],
#     'splitter': ['best'],
#     'max_depth': [4],
#     'min_samples_split': [0.20],
#     'min_samples_leaf': [0.20],
#     'max_features': ['sqrt'],
#     'ccp_alpha': [0.1]  # appears to be costly parameter
# }

# RandomForestClassifier_params = {
#     'criterion': ['gini'],
#     # with this undefined, but criterion full set and class_weight as balanced, the result is better
#     'max_depth': [4],
#     'min_samples_split': [0.20],
#     'min_samples_leaf': [0.20],
#     'max_features': ['sqrt'],
#     'class_weight': ['balanced'],
#     'ccp_alpha': [0.1]
# }


# GradientBoostingClassifier_params = {
#     'loss': ['log_loss'],
#     'learning_rate': [0.1],
#     'n_estimators': [1000],
#     'criterion': ['squared_error'],
#     'min_samples_split': [0.20],
#     'min_samples_leaf': [0.20],
#     'max_depth': [4],
#     'max_features': ['sqrt'],
#     'ccp_alpha': [0.1]
# }
