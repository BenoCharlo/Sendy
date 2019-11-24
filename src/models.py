import numpy as np
import pandas as pd
from time import time
import pprint
import joblib

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool, cv

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

        y = target.values

        fold = 0
        scores = []
        for train_index, test_index in kf.split(data):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = self.train_knn(X_train, y_train, n_neighbors)
            y_pred = model.predict(X_test)
            scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            fold += 1

        return scores

    def tune_knn_cv(self, data, target, kf, max_neigbors):

        knn_scores = []
        for neighbors in list(np.power(2, [x for x in range(1, max_neigbors)])):
            knn_scores.append(np.mean(self.train_knn_cv(data, target, kf, neighbors)))

        scores = pd.DataFrame(
            {"neighbors": [n for n in range(1, max_neigbors)], "rmse_knn": knn_scores,}
        )

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


class CatBoost_Model:
    def __init__(self):
        return None

    def train_catboost(self, data, target, params):
        bst = CatBoostRegressor(
            params["iterations"],
            params["learning_rate"],
            params["depth"],
            random_seed=42,
            logging_level="Silent",
        )

        if aliases.order_index[0] in list(data.columns):
            data.drop(aliases.order_index, axis=1, inplace=True)

        bst.fit(
            data, target, plot=False,
        )
        return bst

    def train_catboost_cv(self, data, target, params, nb_fold):
        if aliases.order_index[0] in list(data.columns):
            data.drop(aliases.order_index, axis=1, inplace=True)

        cv_data = cv(Pool(data, target), params, fold_count=nb_fold, plot=False)
        return cv_data

    def predict_catboost(self, bst, data):
        return bst.predict(data)


def report_perf(optimizer, data, target, title, callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers
    
    optimizer = a sklearn or a skopt optimizer
    data = the training set 
    target = our target
    title = a string label for the experiment
    """
    start = time()
    if callbacks:
        optimizer.fit(data, target, callback=callbacks)
    else:
        optimizer.fit(data, target)
    best_score = optimizer.best_score_
    best_score_std = optimizer.cv_results_["std_test_score"][optimizer.best_index_]
    best_params = optimizer.best_params_
    print(
        (
            title
            + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
            + u"\u00B1"
            + " %.3f"
        )
        % (
            time() - start,
            len(optimizer.cv_results_["params"]),
            best_score,
            best_score_std,
        )
    )
    print("Best parameters:")
    pprint.pprint(best_params)
    print()
    return best_params
