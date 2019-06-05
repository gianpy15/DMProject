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

best_MAE=100

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
        X = pd.DataFrame(X)

        for c in X.columns:
            try:
                X[c]=X[c].astype('float')
            except ValueError:
                X[c]=X[c].astype('category')
        preds = self.model.predict(X)
        return preds


    def validate(self):
    #TODO: DO NOT DELETE IS USEFULL FOR FINISH THE FIT METHOD IN CASE OF VALIDATION

        def _hera_callback(param):
            iteration_num = param[2]
            if iteration_num % param[1]['print_every'] == 0:
                message = f'PARAMS:\n'
                for k in param[1]:
                    message += f'{k}: {param[1][k]}\n'
                Melissa.send_message(f'ITERATION_NUM: {iteration_num}\n {message}\n MRR: {param[5][0][2]}', account='edo')

        # initialize the model
        self.model = lgb.LGBMRanker(**self.params_dict)

        self.model.fit(self.x_train, self.y_train, group=self.groups_train, eval_set=[(self.x_vali, self.y_vali)],
                  eval_group=[self.groups_vali], eval_metric=_mrr, eval_names='validation_set',
                  early_stopping_rounds=200, verbose=False, callbacks=[eval_callback, _hera_callback])
        # save the model parameters
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
        check_folder(f'{self._BASE_PATH}/{time}')
        with open(f"{self._BASE_PATH}/{time}/Parameters.txt", "w+") as text_file:
            text_file.write(str(self.params_dict))
        self.model.booster_.save_model(f'{self._BASE_PATH}/{time}/{self.name}')
        # return negative mrr
        return self.eval_res['validation_set']['MRR'][self.model.booster_.best_iteration - 1]

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
    """
    weather_cols = [f'WEATHER_{i}' for i in range(-10, 0)]
    categorical_cols = ['EMERGENCY_LANE', 'ROAD_TYPE', 'EVENT_DETAIL', 'EVENT_TYPE'] + weather_cols
    X, y = data.dataset('train', onehot=False)
    #X.fillna(0, inplace=True)
    X = reduce_mem_usage(X)

    X_test, y_test =data.dataset('test', onehot=False)
    #X_test.fillna(0, inplace=True)
    X_test=reduce_mem_usage(X_test)
    #y_test=reduce_mem_usage(y_test)


    params_dict = {
        'objective': 'regression_l1',
        'boosting_type':'goss',
        'num_leaves': 21,
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
        #'categorical_feature':'name:'+','.join(categorical_cols),
    }

    params_dict['X'] = X
    model = lightGBM(params_dict=params_dict)
    model_wrapper = MultiOutputRegressor(model)
    model_wrapper.fit(X, y)
    #print(evaluate(model_wrapper, X_test, y_test))
    """
    opt = OptimizerWrapper(lightGBM)
    opt.optimize_random()














