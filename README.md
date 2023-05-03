# Introduction
This project utilises Airbnb listings data to identify optimum `regression` model and its hyperparameters to predict an Airbnb listing's price per night based on numeric data such as number of guests, various ratings, number of bedrooms etc. Mimimum `RMSE` is currently applied to identify the optimum model. Similarly, optimum `classification` model is identified to predict Category of a listing based on the highest `accuracy` score.

`neural network` model is also utilised and tuned for optimum hyperparameters to (similarly to `regression`) predict price per night.

# Regression
`Regression` modelling includes:

1. Implementing a custom function that tunes hyperparameters of various regression models.
2. Carrying out the same task using sklearn `GridSearchCV`.
3. Running hyperparameter tuning on different regression models, namely `SGDRegressor`, `LinearRegression`, `RandomForestRegressor`, `DecisionTreeRegressor`, and `GradientBoostingRegressor` in order to identify (1) optimum set of hyperparameters for each regression model and (2) performance metrics for each of these models with their optimum hyperparameters. Performance metrics used here are `R^2` and `RMSE`, but for the time being `RMSE` is utilised to select the best model. Please note that this is simplistic.
4. The hyperparameter-tuned model and their corresponding hyperparameters and metrics are saved in a local folder.
5. The best model is then loaded into the file alongside its hyperparameters and metrics.

# Classification
`Classification` modelling follows the same process as `Regression` except the models used namely: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, and `GradientBoostingClassifier`.

# Code Organizaton
The code is organized into three main folders, /config, /utils, and /scripts.

## /config
Includes information on hyperparameters used for tuning `Regression`, `Classification`, and `Neural Network` models as well as column grouping specific to the Airbnb data useed in this project.

## /utils
Comprises two files. `data_utils.py` includes functions used to clean, load, and split data used in the project. `model_utils.py` includes functions used to custom tune hyperparameters of different models, save the optimum models (based on a chosen metric) for each regressor / classifier and their corresponding hyperparameters and metrics and load in a model based on the optimum chosen metric.

# /scripts
This folder includes three files `regression.py` used to execute the optimization of regression code; `classification.py` used in the same manner; and `neural_network.py`.



