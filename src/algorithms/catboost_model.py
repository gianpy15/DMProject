import sys
import os
sys.path.append(os.getcwd())

from src.algorithms.chainable_model import ChainableModel
from catboost import CatBoostRegressor

class CatBoost(ChainableModel):

    def build_model(self, params_dict):
        self.name = 'catboost'
        return CatBoostRegressor(**params_dict)

    def fit_model(self, X, y, X_val, y_val):
        self.model.fit(X, y, eval_set=[(X_val, y_val)])

    def predict(self, X):
        preds = self.model.predict(X)
        return preds


if __name__ == "__main__":
    import src.data as data
    import numpy as np

    from src.algorithms.multioutput import MultiOutputRegressor, RegressorChain
    from sklearn.utils import shuffle
    from src.algorithms.catboost_model import CatBoost
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    print()
    chain_mode = input('Choose the chain mode (m: multioutput / c: regressorchain): ').lower()
    M = MultiOutputRegressor if chain_mode == 'm' else RegressorChain

    X, Y = data.dataset(onehot=False, drop_index_columns=True)

    X.fillna(0, inplace=True)

    weather_cols = [f'WEATHER_{i}' for i in range(-10,0)]
    categorical_cols = ['EMERGENCY_LANE', 'ROAD_TYPE', 'EVENT_DETAIL','EVENT_TYPE'] + weather_cols

    catboost = CatBoost({
        'X': X,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'n_estimators':20,
        'depth':6,
        'learning_rate':1,
        'early_stopping_rounds': 15,
        'cat_features': categorical_cols
    })

    model = M(catboost)
    model.fit(X, Y)

    def evaluate(X_test, y_test):
        mask_test = np.all(y_test.notnull(), axis=1)

        y_pred = model.predict(X_test[mask_test])
        return mean_absolute_error(y_test[mask_test], y_pred)


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    mae = evaluate(X_test, y_test)
    print(mae)
