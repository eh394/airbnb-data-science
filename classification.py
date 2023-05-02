import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import data_config
import data_utils
import model_config
import model_utils


np.random.seed(2)


df = data_utils.load_df(
    "listings.csv",
    "listings_clean.csv",
    data_config.rating_columns,
    "Description",
    data_config.default_value_columns,
    1
)

X_train, y_train, X_validation, y_validation, X_test, y_test = data_utils.load_split_X_y(
    df, data_config.feature_columns, "Category", 0.7, 0.5)


epochs = 1000


models = [
    (LogisticRegression, model_config.LogisticRegression_params),
    (DecisionTreeClassifier, model_config.DecisionTreeClassifier_params),
    (RandomForestClassifier, model_config.RandomForestClassifier_params),
    (GradientBoostingClassifier, model_config.GradientBoostingClassifier_params)
]


if __name__ == "__main__":

    model_utils.evaluate_all_models(
        models,
        X_train,
        y_train,
        X_validation,
        y_validation,
        model_config.classification_metrics,
        model_utils.derive_metrics_classification,
        "models/classification",
        "accuracy"
    )

    opt_model_name, opt_metric = model_utils.find_optimal_model(
        models,
        "models/classification",
        "accuracy"
    )

    opt_model, opt_params, opt_metrics = model_utils.load_model(
        "models/classification",
        opt_model_name
    )

    print(opt_model, opt_params, opt_metrics)



