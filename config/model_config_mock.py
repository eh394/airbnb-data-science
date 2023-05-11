regression_metrics = ['R_sq', 'RMSE']
classification_metrics = ['accuracy', 'precision', 'recall', 'f1']

maximise_metrics = ['R_sq', 'accuracy', 'precision', 'recall', 'f1']
minimise_metrics = ['RMSE']

SGDRegressor_params = {
    'loss': ['squared_error'],
    'penalty': ['l2'],
    'learning_rate': ['optimal'],
    'alpha': [0.01],
    'max_iter': [5000],
    'early_stopping': [False]
}

LinearRegression_params = {
    'fit_intercept': [True],
    'copy_X': [True],
    'n_jobs': [None],
    'positive': [False]
}

DecisionTreeRegressor_params = {
    'criterion': ['squared_error'],
    'splitter': ['best'],
    'max_depth': [3],
    'min_samples_split': [0.20],
    'min_samples_leaf': [0.20],
    'max_features': ['sqrt'],
    'ccp_alpha': [0.01]
}

RandomForestRegressor_params = {
    'n_estimators': [100],
    'criterion': ['squared_error'],
    'max_depth': [4],
    'min_samples_split': [0.20],
    'min_samples_leaf': [0.20],
    'max_features': ['sqrt'],
    'bootstrap': [False],
    'ccp_alpha': [0.01]
}

GradientBoostingRegressor_params = {
    'loss': ['squared_error'],
    'learning_rate': [0.1],
    'n_estimators': [100],
    'criterion': ['squared_error'],
    'min_samples_split': [0.20],
    'min_samples_leaf': [0.20],
    'max_depth': [4],
    'max_features': ['sqrt']
}

LogisticRegression_params = {
    'penalty': ['l2'],
    'class_weight': ['balanced'],
    'solver': ['saga'],
    'max_iter': [5000],
    'multi_class': ['multinomial']
}

DecisionTreeClassifier_params = {
    'criterion': ['gini'],
    'splitter': ['best'],
    'max_depth': [4],
    'min_samples_split': [0.20],
    'min_samples_leaf': [0.20],
    'max_features': ['sqrt'],
    'ccp_alpha': [0.1]
}

RandomForestClassifier_params = {
    'criterion': ['gini'],
    'max_depth': [4],
    'min_samples_split': [0.20],
    'min_samples_leaf': [0.20],
    'max_features': ['sqrt'],
    'class_weight': ['balanced'],
    'ccp_alpha': [0.1]
}

GradientBoostingClassifier_params = {
    'loss': ['log_loss'],
    'learning_rate': [0.1],
    'n_estimators': [1000],
    'criterion': ['squared_error'],
    'min_samples_split': [0.20],
    'min_samples_leaf': [0.20],
    'max_depth': [4],
    'max_features': ['sqrt'],
    'ccp_alpha': [0.1]
}
