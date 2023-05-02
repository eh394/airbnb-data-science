import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor

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
    df, data_config.feature_columns, "Price_Night", 0.7, 0.5)

epochs = 1000


models = [
    (SGDRegressor, model_config.SGDRegressor_params),
    (LinearRegression, model_config.LinearRegression_params),
    (DecisionTreeRegressor, model_config.DecisionTreeRegressor_params),
    (RandomForestRegressor, model_config.RandomForestRegressor_params),
    (GradientBoostingRegressor, model_config.GradientBoostingRegressor_params)
]


if __name__ == "__main__":
    model_utils.evaluate_all_models(
        models,
        X_train,
        y_train,
        X_validation,
        y_validation,
        model_config.regression_metrics,
        model_utils.derive_metrics_regression,
        "models/regression",
        "RMSE"
    )

    opt_model_name, opt_metric = model_utils.find_optimal_model(
        models,
        "models/regression",
        "RMSE"
    )

    opt_model, opt_params, opt_metrics = model_utils.load_model(
        "models/regression",
        opt_model_name
    )

    print(opt_model, opt_params, opt_metrics)



