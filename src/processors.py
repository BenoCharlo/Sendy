import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# from sklearn.pipeline import Pipeline

from src import aliases


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified,  transforms all
        columns in X.
        """
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class Preprocessor:
    def __init__(self, data=None):
        return None

    def separate(self, data):
        target = data["Time from Pickup to Arrival"]
        data.drop("Time from Pickup to Arrival", axis=1)

        return data, target

    def remove_na_variable(self, data):
        return data.drop(["Precipitation in millimeters"], axis=1)

    def drop_variables(self, data, train_type=True):
        if train_type:
            data = data.drop(aliases.to_drop, axis=1)
        else:
            to_drop = aliases.to_drop
            to_drop = [
                variable
                for variable in to_drop
                if variable != "Arrival at Destination - Time"
            ]
            data = data.drop(to_drop, axis=1)

        return data

    def le_matrix(self, data):
        data_categorical = data.drop(aliases.not_to_encoded, axis=1)
        data_categorical = MultiColumnLabelEncoder().fit_transform(data_categorical)
        data = pd.concat(
            [data_categorical, data.filter(aliases.not_to_encoded, axis=1)], axis=1
        )
        return data

    def ohe_matrix(self, data):
        encoder = OneHotEncoder(handle_unknown="ignore")

        data_categorical = data.filter(aliases.to_categorical, axis=1)
        data_categorical = encoder.fit_transform(data_categorical)
        data_categorical = data_categorical.toarray()
        data = pd.concat(
            [data_categorical, data.filter(aliases.not_to_encoded, axis=1)], axis=1
        )
        return data

    def preprocess_data(self, data, train_type):

        preprocessed_data = self.le_matrix(
            self.drop_variables(self.remove_na_variable(data), train_type)
        )
        return preprocessed_data
