# Introduction
An exploration of Airbnb listing data and python ML libraries to create a number of prediction models. The optimum `Regression` model and its hyperparameters are identified, to predict an Airbnb listing's price per night based on numeric data such as number of guests, various ratings, number of bedrooms etc. Minimum `RMSE` is applied to identify the optimum model and its corresponding hyperparameters. Similarly, the optimum `Classification` model is identified to predict a category of a listing based on the highest `accuracy` score. A `Neural Network` model is also utilised and tuned for optimum hyperparameters to (similarly to `Regression`) predict price per night.


# Regression
`Regression` modelling involves running hyperparameter tuning on different regression models, namely `SGDRegressor`, `LinearRegression`, `RandomForestRegressor`, `DecisionTreeRegressor`, and `GradientBoostingRegressor` in order to identify (1) optimum set of hyperparameters for each regression model and (2) their corresponding performance metrics. Performance metrics used here are `R^2` and `RMSE`, minimum `RMSE` is utilised to select the model. 


# Classification
`Classification` modelling follows the same process as `Regression` except the models used, namely: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, and `GradientBoostingClassifier`; and metrics employed, i.e. `accuracy`, `precision`, `recall` and `f1 score`. `accuracy` is used to identify the optimum model and its corresponding hyperparameters.

# Code Organizaton
The code is organized into three main folders, `config`, `utils`, and `scripts`. Note that the scripts must be ran as modules from the project root directory. For example:<br>
`python -m scripts.regression` / add new line

## config
Contains information on hyperparameters used for tuning `Regression`, `Classification`, and `Neural Network` models as well as column grouping specific to the Airbnb dataset.

## utils
`data_utils.py` contains functions used to clean, load, and split Airbnb data. `model_utils.py` contains functions used to (1) custom tune hyperparameters of different models, (2) save the optimum models (based on a chosen metric) for each regressor / classifier alongside their corresponding hyperparameters and metrics, and (3) load in the optimum model.


# Further Work
Suggestions for further work include:
1. Utilise a larger dataset to produce models with better metrics, for instance current R^2 in the range of 0.60 for regression price prediction model could be improved upon.
2. Incorporate regularization into the code.
3. Incorporate reporting into the project, namely metrics on the test dataset as well as visual representation of the results.




