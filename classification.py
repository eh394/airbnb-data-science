import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import data_config
import data_utils
import model_config
import model_utils


np.random.seed(2)


epochs = 1000


models = [
    (LogisticRegression, model_config.LogisticRegression_params),
    (DecisionTreeClassifier, model_config.DecisionTreeClassifier_params),
    (RandomForestClassifier, model_config.RandomForestClassifier_params),
    (GradientBoostingClassifier, model_config.GradientBoostingClassifier_params)
]


if __name__ == "__main__":

    df = data_utils.load_df(
        raw_data_filename="listings.csv",
        clean_data_filename="listings_clean.csv",
        missing_values_subset=data_config.rating_columns,
        description_string_subset="Description",
        default_values_subset=data_config.default_value_columns,
        default_value=1
    )

    X_train, y_train, X_validation, y_validation, X_test, y_test = data_utils.load_split_X_y(
        df,
        features=data_config.feature_columns,
        labels="Category",
        train_test_proportion=0.7,
        test_validation_proportion=0.5
    )

    model_utils.evaluate_all_models(
        models,
        X_train,
        y_train,
        X_validation,
        y_validation,
        model_config.classification_metrics,
        model_utils.derive_metrics_classification,
        output_folder="models/classification",
        config_metric="accuracy"
    )

    opt_model_name, opt_metric = model_utils.find_optimal_model(
        models,
        output_folder="models/classification",
        config_metric="accuracy"
    )

    opt_model, opt_params, opt_metrics = model_utils.load_model(
        opt_model_name,
        folder="models/classification"
    )

    print(opt_model, opt_params, opt_metrics)
