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
    import src.utils.folder as folder
    import src.algorithms.inout as inout

    from src.algorithms.multioutput import MultiOutputRegressor, RegressorChain
    from sklearn.utils import shuffle
    from src.algorithms.catboost_model import CatBoost
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    print()
    chain_mode = input('Choose the chain mode (m: multioutput / c: regressorchain): ').lower()
    M = MultiOutputRegressor if chain_mode == 'm' else RegressorChain

    #X, Y = data.dataset_with_features('train', onehot=False, drop_index_columns=True)
    X, Y = data.dataset('train', onehot=False)

    X.fillna(0, inplace=True)

    weather_cols = [col for col in X.columns if col.startswith('WEATHER_')]
    categorical_cols = ['EMERGENCY_LANE', 'ROAD_TYPE', 'EVENT_DETAIL','EVENT_TYPE'] + weather_cols

    catboost = CatBoost({
        'X': X,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'n_estimators':3500,
        'depth':6,
        'learning_rate':1,
        'early_stopping_rounds': 100,
        'cat_features': categorical_cols
    })
    
    model = M(catboost)
    model.fit(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    mae = inout.evaluate(model, X_test, y_test)
    print()
    print(mae)

    # save the model
    mae = round(mae, 5)

    suffix = input('Insert model name suffix: ')
    model_folder = 'saved_models'
    folder.create_if_does_not_exist(model_folder)

    chain_mode = 'chain' if chain_mode == 'c' else 'multiout'
    filename = f'catboost_{chain_mode}_{mae}_{suffix}.jl'
    inout.save(model, os.path.join(model_folder, filename))