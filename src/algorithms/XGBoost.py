import sys
import os
sys.path.append(os.getcwd())

from src.algorithms.chainable_model import ChainableModel
import xgboost as xgb



class XGBoost(ChainableModel):

    def build_model(self, params_dict):
        self.name = 'xgboost'
        return xgb.XGBRegressor(**params_dict)

    def fit_model(self, X, y, X_val, y_val):
        self.model.fit(X, y, eval_set=[(X_val, y_val)])

    def predict(self, X):
        preds = self.model.predict(X)
        return preds




if __name__ == '__main__':
    
    import src.data as data
    import numpy as np

    from src.algorithms.multioutput import MultiOutputRegressor, RegressorChain
    from sklearn.utils import shuffle
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    print()
    chain_mode = input('Choose the chain mode (m: multioutput / c: regressorchain): ').lower()
    M = MultiOutputRegressor if chain_mode == 'm' else RegressorChain

    X, Y = data.dataset(onehot=False, drop_index_columns=True)

    # add features
    import src.preprocessing.other_features as feat
    avg_speed_road_event = feat.avg_speed_for_roadtype_event()
    X = X.merge(avg_speed_road_event, how='left', on=['EVENT_TYPE','ROAD_TYPE'])
    del avg_speed_road_event

    X.fillna(0, inplace=True)

    weather_cols = [f'WEATHER_{i}' for i in range(-10,0)]
    categorical_cols = ['EMERGENCY_LANE', 'ROAD_TYPE', 'EVENT_DETAIL','EVENT_TYPE'] + weather_cols

    xgboost = XGBoost({
        'X':X
        'objective' :'reg:linear',
        'colsample_bytree' : 0.3,
        'learning_rate' : 0.1,
        'max_depth' : 5,
        'alpha' : 10,
        'n_estimators' : 10
    })

    model = XGBoost(xgboost)
    model_wrapper = MultiOutputRegressionChainWrapper(model, X, y)
    model_wrapper.fit()

    def evaluate(X_test, y_test):
        mask_test = np.all(y_test.notnull(), axis=1)

        y_pred = model.predict(X_test[mask_test])
        return mean_absolute_error(y_test[mask_test], y_pred)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    mae = evaluate(X_test, y_test)
    print()
    print(mae)