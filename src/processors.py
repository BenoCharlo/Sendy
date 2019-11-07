from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


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
        return train.drop(["Precipitation in millimeters"], axis=1, inplace=True)

