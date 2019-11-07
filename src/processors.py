from functools import reduce
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import aliases


def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), reversed(functions), lambda x: x)


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
    def __init__(self):
        return self

    def separate(data):
        target = train["Time from Pickup to Arrival"]
        data.drop("Time from Pickup to Arrival", axis=1, inplace=True)

        return data, target

    def remove_Na_variable(data):
        return data.drop(["Precipitation in millimeters"], axis=1, inplace=True)

    def drop_variables(data):
        return data.drop(to_drop, axis=1, inplace=True)

    def ohe_matrix(data):
        data_categorical = data.drop(not_to_encoded, axis=1)
        data_categorical = MultiColumnLabelEncoder().fit_transform(data_categorical)
        data = pd.concat(
            [data_categorical, data.filter(not_to_encoded, axis=1)], axis=1
        )

    def preprocess_data(data):

        preprocessed_data = compose(
            lambda data: remove_Na_variables(data),
            lambda data: drop_variables(data),
            lambda data: ohe_matrix(data),
        )(data)

    return preprocess_data
