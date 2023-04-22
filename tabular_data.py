from ast import literal_eval
import pandas as pd
import numpy as np
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
    df[subset] = df[subset].replace(np.nan, 1)
    return df


def clean_tabular_data(df):
    df = remove_rows_with_missing_ratings(df, rating_columns)
    df = combine_description_strings(df, 'Description')
    df = set_default_feature_values(df, default_value_columns, 1)
    return df


# add a provision to make sure this does not happen if file already exists
# if __name__ == "__main__":
#     df = pd.read_csv('listing.csv')
#     df = clean_tabular_data(df)
#     df.to_csv('clean_tabular_data.csv')


def load_airbnb(df, subset, label):
    features = df[subset]
    labels = df[label]
    return (features, labels)

# add if name main blocks into all files

