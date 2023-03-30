import pandas as pd
import numpy as np
from tabular_data import load_airbnb, rating_columns, default_value_columns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import math

np.random.seed(2)

df = pd.read_csv('clean_tabular_data.csv')
features, labels = load_airbnb(df, (rating_columns + default_value_columns), 'Price_Night') 

X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, test_size=0.3)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5)

# your_model = LinearRegression()
# your_model.fit(x_train, y_train)
# print(your_model.coef_)
# print(your_model.intercept_)
# print(your_model.score(x_test, y_test))
epochs = 10000
your_model = SGDRegressor() # max_iter=1000
your_model.fit(X_train, y_train)
print(your_model.coef_)
print(your_model.intercept_)
predicted_y_train = your_model.predict(X_train)
predicted_y_test = your_model.predict(X_test)
MSE_train = mean_squared_error(y_train, predicted_y_train)
MSE_test = mean_squared_error(y_test, predicted_y_test)
RMSE_train = math.sqrt(MSE_train)
RMSE_test = math.sqrt(MSE_test)
print(f"R^2 for the training data: {your_model.score(X_train, y_train)}")
print(f"RMSE for the trainining data: {RMSE_train}")
print(f"R^2 for the test data: {your_model.score(X_test, y_test)}")
print(f"RMSE for the test data: {RMSE_test}")


# print(len(X_train))
# print(len(X_validation))
# print(len(X_test))

def custom_tune_regression_model_hyperparameters(
        model_class,
        X_train,
        y_train
        X_validation,
        y_validation,
        X_test,
        y_test
        hyperparameters_dictionary):
    pass
