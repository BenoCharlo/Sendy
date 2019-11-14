import xgboost as xgb
import lightgbm as lgb

# from sklearn

from src import aliases


class XGB_Model:
    def __init__(self):
        return None

    def prepare_data(self, data, target=None):
        if aliases.order_index in list(data.columns):
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
        if aliases.order_index in list(data.columns):
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
