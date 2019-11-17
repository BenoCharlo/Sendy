from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

from src import aliases


class KFold_Strategy:
    def __init__(self):
        return None

    def kfold_split(self, data, n_splits):
        kf = KFold(n_splits=n_splits, random_state=43)

        return kf


class KNN_Model:
    def __init__(self):
        return None

    def prepare_data(self, data):
        if aliases.order_index[0] in list(data.columns):
            data.drop(aliases.order_index, axis=1, inplace=True)

        return data.values

    def train_knn(self, data, target, n_neighbors):
        model = KNeighborsRegressor(n_neighbors)

        model.fit(data, target)
        return model

    def predict_knn(self, knn_model, data):
        return model.predict(data)

    def train_knn_cv(self, data, target, kf, n_neighbors):
        # assert(0, kf.get_n_splits)
        y = target.values

        fold = 0
        scores = []
        for train_index, test_index in kf.split(data):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = self.train_knn(X_train, y_train, n_neighbors)
            y_pred = model.predict(X_test)
            print(f"====== Fold {fold} ======")
            print(mean_squared_error(y_test, y_pred))
            scores.append(mean_squared_error(y_test, y_pred))
            fold = +1

        return scores


class XGB_Model:
    def __init__(self):
        return None

    def prepare_data(self, data, target=None):
        if aliases.order_index[0] in list(data.columns):
            data.drop(aliases.order_index, axis=1, inplace=True)

        if target is None:
            data = xgb.DMatrix(data, label=target)
        else:
            data = xgb.DMatrix(data, label=target)
        return data

    def train_xgb(self, data, params, num_boost_round):
        bst = xgb.train(params, data, num_boost_round)
        return bst

    def train_xgb_cv(self, data, params, nfold, num_boost_round):

        cv_rmse_xgb = xgb.cv(
            params,
            data,
            num_boost_round=num_boost_round,
            nfold=nfold,
            stratified=False,
            folds=None,
            metrics="rmse",
            seed=43,
        )
        return cv_rmse_xgb

    def predict_xgb(bst, data):
        return bst.predict(data)


class LGB_Model:
    def __init__(self):
        return None

    def prepare_data(self, data, target=None):
        if aliases.order_index[0] in list(data.columns):
            data.drop(aliases.order_index, axis=1, inplace=True)

        if target is None:
            data = lgb.Dataset(data, label=target)
        else:
            data = lgb.Dataset(data, label=target)
        return data

    def train_lgb(self, data, params, num_boost_round):
        bst = lgb.train(params, data, num_boost_round)
        return bst

    def train_lgb_cv(self, data, params, nfold, num_boost_round):

        cv_rmse_lgb = lgb.cv(
            params,
            train_set=data,
            num_boost_round=num_boost_round,
            nfold=nfold,
            stratified=False,
            folds=None,
            metrics="rmse",
            seed=43,
        )
        return cv_rmse_lgb

    def predict_lgb(bst, data):
        return bst.predict(data)


# class STACKING_Model():
#     def __init__(self):
#         return None

#     def get_kfold():
