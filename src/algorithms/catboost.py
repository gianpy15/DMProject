from src.algorithms.chainable_model import ChainableModel
from catboost import CatBoostRegressor

class CatBoost(ChainableModel):

    def build_model(self, params_dict):
        self.model_name = 'catboost'
        return CatBoostRegressor(**params_dict)

    def fit_model(self, X, y, X_val, y_val):
        self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=True)

    def predict(self, X):
        preds = self.model.predict(X)
        return preds
