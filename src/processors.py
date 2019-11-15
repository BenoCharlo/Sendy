import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

    def separate_train_target(self, data):
        target = data["Time from Pickup to Arrival"]
        data.drop("Time from Pickup to Arrival", axis=1)

        return data, target

    def change_var_type(self, data):
        assert set(aliases.to_categorical).issubset(list(data.columns))
        assert set(aliases.to_datetime).issubset(list(data.columns))

        for col in aliases.to_categorical:
            data[col] = data[col].astype("category")

        for col in aliases.to_datetime:
            data[col] = pd.to_datetime(data[col])

        return data

    def remove_na_variable(self, data):
        return data.drop(["Precipitation in millimeters"], axis=1)

    def drop_variables(self, data, is_train=True):
        if is_train:
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

    def join_train_test(self, train_data, test_data):
        test_cols = list(test_data.columns)
        train_data = train_data[test_cols]

        data = pd.concat([train_data, test_data], axis=0)

        data["is_train"] = [1] * train_data.shape[0] + [0] * test_data.shape[0]

        return data

    def preprocess_data(self, data, is_train):

        preprocessed_data = self.drop_variables(
            self.remove_na_variable(self.change_var_type(data)), is_train
        )
        return preprocessed_data

    def le_matrix(self, data):
        data_categorical = data.drop(aliases.not_to_encoded, axis=1)
        data_categorical = MultiColumnLabelEncoder().fit_transform(data_categorical)
        data = pd.concat(
            [data_categorical, data.filter(aliases.not_to_encoded, axis=1)], axis=1
        )
        return data

    def ohe_matrix(self, data):
        encoder = OneHotEncoder(handle_unknown="ignore")

        data_ohe = encoder.fit_transform(data[aliases.to_categorical])
        data_categorical = np.concatenate(
            [data_ohe.toarray(), data.drop(aliases.to_categorical, axis=1)], axis=1
        )

        cols = list(encoder.get_feature_names()) + list(
            data.drop(aliases.to_categorical, axis=1).columns
        )

        data = pd.DataFrame(data_categorical, columns=cols)

        return data.drop(aliases.not_to_encoded, axis=1)

    def separate_train_test(self, data):
        assert "is_train" in list(data.columns)

        train_data = data.loc[data["is_train"] == 1]
        test_data = data.loc[data["is_train"] == 0]

        return train_data.drop("is_train", axis=1), test_data.drop("is_train", axis=1)

