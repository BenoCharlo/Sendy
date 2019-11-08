import xgboost as xgb


class XGB_Model:
    def __init__(self):
        return None

    def prepare_data(self, data, target=None):
        if target is None:
            data = xgb.DMatrix(data.drop("Order No", axis=1), label=target)
        else:
            data = xgb.DMatrix(data.drop("Order No", axis=1), label=target)
        return data

    def train_xgb(self, data, params, num_boost_round):
        bst = xgb.train(params, dtrain, num_boost_round)
        return bst

    def train_xgb_cv(dself, ata, params, nfold, num_boost_round):

        cv_rmse_xgb = xgb.cv(
            params,
            dtrain,
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
