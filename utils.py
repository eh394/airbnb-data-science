


hyperparams_dict = {
    'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet', None],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    'max_iter': [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
    'early_stopping': [True, False]
}