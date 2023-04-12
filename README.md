# Milestone 3
Please note that Milestones 1 and 2 involved setting up the environment and getting familiar with the overall scope of the project. Milestone 3 involved: 

1. Writing a series of simple functions in `tabular_data.py` file to load in the airbnb data from a csv file, carry out cleaning operations on the data and then save them to another csv.file. 
2. Another function `load_airbnb` loads numeric columns of this dataset as pandas dataframe `features` and price per night column as `labels` in preparation for data analysis.

# Milestone 4
Milestone 4 involved:

1. Implementing a custom function that tunes hyperparameters of various regression models.
2. Carrying out the same task using sklearn `GridSearchCV`.
3. Running hyperparameter tuning on different regression models, namely `SGDRegressor`, `LinearRegression`, `RandomForestRegressor`, `DecisionTreeRegressor`, and `GradientBoostingRegressor` in order to identify (1) optimum set of hyperparameters for each regression model and (2) performance metrics for each of these models with their optimum hyperparameters. Performance metrics used here are `R^2` and `RMSE`, but for the time being `RMSE` is utilised to select the best model. Please note that this is simplistic.
4. The hyperparameter-tuned model and their corresponding hyperparameters and metrics are saved in a local folder.
5. The best model is then loaded into the file alongside its hyperparameters and metrics.

# Milestone 5
Milesstone 5 involved:

1. Similarly to Milestone 4, implementing a custom function that tunes hyperparameters of various classification models.
2. As above.
3. As above for `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, and `GradientBoostingClassifier`.
4. As above.
5. As above.


