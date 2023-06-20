import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor

from config import data_config, model_config
from lib import data_utils, model_utils

np.random.seed(2)

epochs = 1000

models = [
    (SGDRegressor, model_config.SGDRegressor_params),
    (LinearRegression, model_config.LinearRegression_params),
    (DecisionTreeRegressor, model_config.DecisionTreeRegressor_params),
    (RandomForestRegressor, model_config.RandomForestRegressor_params),
    (GradientBoostingRegressor, model_config.GradientBoostingRegressor_params)
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
        labels="Price_Night",
        train_test_proportion=0.7,
        test_validation_proportion=0.5
    )

    model_utils.evaluate_all_models(
        models,
        X_train,
        y_train,
        X_validation,
        y_validation,
        model_config.regression_metrics,
        model_utils.derive_metrics_regression,
        output_folder="models/regression",
        config_metric="RMSE"
    )

    opt_model_name, opt_metric = model_utils.find_optimal_model(
        models,
        output_folder="models/regression",
        config_metric="RMSE"
    )

    opt_model, opt_params, opt_metrics = model_utils.load_model(
        opt_model_name,
        folder="models/regression"
    )

    print(opt_model, opt_params, opt_metrics)
