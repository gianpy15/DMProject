import lightgbm as lgb
from tqdm import tqdm
import src.data as data
tqdm.pandas()
from src.algorithms.multioutput import MultiOutputRegressor
from src.algorithms.multioutput import RegressorChain
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
from skopt.space import Real, Integer, Categorical
from src.utility import reduce_mem_usage
from src.algorithms.chainable_model import ChainableModel
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import src.utils.telegram_bot as Melissa
from src.Optimizer import OptimizerWrapper
import src.utils.exporter as exporter

best_MAE=100

def to_cat(df):
    weather_cols = [col for col in df.columns if col.startswith('WEATHER_')]
    categorical_cols = ['EMERGENCY_LANE', 'ROAD_TYPE', 'EVENT_DETAIL', 'EVENT_TYPE',
                        'WEEK_DAY', 'IS_WEEKEND'] + weather_cols
    for c in categorical_cols:
        df[c] = df[c].astype('category')
    return df


def evaluate(model, X_test, y_test):
    mask_test = np.all(y_test.notnull(), axis=1)
    print('number of valid samples:', (mask_test*1).sum())

    y_pred = model.predict(X_test[mask_test])

    """
    for i in range(4):
        print(f'MAE columns {i}\n')
        print(mean_absolute_error(y_test[mask_test].values[:, i], y_pred[:, i]))
        print('\n')
    """
    return mean_absolute_error(y_test[mask_test], y_pred)


class lightGBM(ChainableModel):

    def build_model(self, params_dict):
        #self.mode = mode
        #self.params_dict = params_dict
        #self.eval_res = {}
        return lgb.LGBMRegressor(**params_dict)

    #def build_model(self, params_dict):
    def fit_model(self, X, y, X_val, y_val):
        self.model.fit(X, y, eval_set=[(X_val, y_val)], eval_metric='regression_l1', verbose=1,
                           eval_names='validation_set', early_stopping_rounds=200)
        #fig, ax = plt.subplots(figsize=(12, 10))
        #a = lgb.plot_importance(self.model.booster_, ax=ax)
        #plt.subplot(a)
        #plt.show()

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.params_dict['X'].columns)\
            .astype(self.col_dtypes)
        #X = to_cat(X)
        preds = self.model.predict(X)
        return preds

    @staticmethod
    def get_optimize_params():
        space = [
            Real(0.01, 0.2, name='learning_rate'),
            Integer(6, 80, name='num_leaves'),
            Real(0, 100, name='reg_lambda'),
            Real(0, 100, name='reg_alpha'),
            Real(0, 10, name='min_split_gain'),
            Real(0, 10, name='min_child_weight'),
            Integer(10, 1000, name='min_child_samples'),
        ]

        def get_MAE(arg_list):
            learning_rate, num_leaves, reg_lambda, reg_alpha, min_split_gain, \
            min_child_weight, min_child_samples = arg_list
            """
            Melissa.send_message(f'Starting a train of bayesyan search with following params:\n '
                              f'learning_rate:{learning_rate}, num_leaves:{num_leaves}, '
                              f'reg_lambda{reg_lambda}, reg_alpha:{reg_alpha}, min_split_gain:{min_split_gain}'
                              f'min_child_weight:{min_child_weight}, min_child_samples:{min_child_samples}')
            """
            params_dict = {
                'boosting_type': 'gbdt',
                'num_leaves': num_leaves,
                'max_depth': -1,
                'n_estimators': 5000,
                'learning_rate': learning_rate,
                'subsample_for_bin': 200000,
                'class_weights': None,
                'min_split_gain': min_split_gain,
                'min_child_weight': min_child_weight,
                'min_child_samples': min_child_samples,
                'subsample': 1,
                'subsample_freq': 0,
                'colsample_bytree': 1,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'random_state': None,
                'n_jobs': -1,
                'silent': False,
                'importance_type': 'split',
                'metric': 'None',
                'print_every': 100,
            }
            X, y = data.dataset(onehot=False)
            X = reduce_mem_usage(X)
            params_dict['X'] = X
            model = lightGBM(params_dict=params_dict)
            model_wrapper = MultiOutputRegressor(model, n_jobs=-1)
            model_wrapper.fit(X, y)
            X_train, X_val,y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

            MAE = evaluate(model_wrapper, X_val, y_val)

            global best_MAE
            if MAE<best_MAE:
                best_MAE=MAE
                Melissa.send_message(f'MAE: {MAE}\n'
                                  f'params:\n'
                                  f'learning_rate:{learning_rate}, num_leaves:{num_leaves}, '
                                  f'reg_lambda{reg_lambda}, reg_alpha:{reg_alpha} , min_split_gain:{min_split_gain}'
                                  f'min_child_weight:{min_child_weight}, min_child_samples:{min_child_samples}')
            return MAE

        return space, get_MAE


if __name__ == '__main__':


    X, y = data.dataset('local', 'train', onehot=False)
    X = to_cat(X)

    X_test, y_test, sub_base_structure = data.dataset('full', 'test', onehot=False, export=True)
    #X_test=to_cat(X_test)

    params_dict = {
        'objective': 'regression_l1',
        'boosting_type':'gbdt',
        'num_leaves': 25,
        'max_depth': -1,
        'learning_rate': 0.01,
        'n_estimators': 1500,
        'subsample_for_bin': 200000,
        'class_weights': None,
        'min_split_gain': 0.0,
        'min_child_weight': 0.01,
        'min_child_samples': 1,
        'subsample': 1.0,
        'subsample_freq': 0,
        'colsample_bytree': 1,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'random_state': None,
        'n_jobs': -1,
        'silent': False,
        'importance_type': 'split',
        'metric': 'None',
    }

    params_dict['X'] = X
    model = lightGBM(params_dict=params_dict)
    model_wrapper = MultiOutputRegressor(model)
    model_wrapper.fit(X, y)

    predictions = model_wrapper.predict(X_test)
    sub = exporter.export_sub(sub_base_structure, predictions)
    #exporter.compute_MAE(sub, y_test)


    """
    sub2 = exporter.export_sub(sub_base_structure, predictions)
    
    dict_scores = {
        'bs':[sub, sub2],
        'weights':[0.5, 0.5]
    }
    sub_hybrid = exporter.hybrid_score(dict_scores)
    exporter.compute_MAE(sub_hybrid, y_test)
    """

    #opt = OptimizerWrapper(lightGBM)
    #opt.optimize_random()














