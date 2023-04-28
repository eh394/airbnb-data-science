from ast import literal_eval
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
pd.options.display.max_columns = None
pd.options.display.max_rows = None


rating_columns = [
    'Cleanliness_rating',
    'Accuracy_rating',
    'Communication_rating',
    'Location_rating',
    'Check-in_rating',
    'Value_rating']

default_value_columns = [
    'guests',
    'beds',
    'bathrooms',
    'bedrooms']


def remove_rows_with_missing_ratings(df, subset):
    df.dropna(inplace=True, subset=subset)
    return df


def convert_string_list_to_string(input_string):
    try:
        output_list = pd.eval(input_string)
        output_string = " ".join(output_list[1:])
        return output_string
    except:
        return np.nan


def combine_description_strings(df, subset):
    df[subset] = df[subset].apply(convert_string_list_to_string)
    df.dropna(inplace=True, subset=subset)
    return df


def set_default_feature_values(df, subset, default):
    df[subset] = df[subset].replace(np.nan, default)
    return df


def clean_tabular_data(
        df,
        missing_values_subset,
        description_string_subset,
        default_values_subset,
        default_value
):
    df = remove_rows_with_missing_ratings(df, missing_values_subset)
    df = combine_description_strings(df, description_string_subset)
    df = set_default_feature_values(df, default_values_subset, default_value)
    return df


def load_df(
        raw_data_filename,
        clean_data_filename,
        missing_values_subset,
        description_string_subset,
        default_values_subset,
        default_value
):
    try:
        df = pd.read_csv(clean_data_filename)
    except:
        df = pd.read_csv(raw_data_filename)
        df = clean_tabular_data(
            df, missing_values_subset, description_string_subset, default_values_subset, default_value)
        df.to_csv(clean_data_filename)

    return df


def load_split_X_y(
        df,
        features,
        labels,
        train_test_proportion=0.7,
        test_validation_proportion=0.5
):

    X = df[features]
    y = df[labels]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_test_proportion)
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_test, y_test, test_size=test_validation_proportion)

    return X_train, y_train, X_validation, y_validation, X_test, y_test


df = load_df('listing.csv', 'clean_tabular_data.csv',
             rating_columns, 'Description', default_value_columns, 1)
X_train, y_train, X_validation, y_validation, X_test, y_test = load_split_X_y(
    df, (rating_columns + default_value_columns), 'Price_Night', 0.7, 0.5)


# if __name__ == "__main__":
