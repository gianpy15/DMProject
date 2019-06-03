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



class lightGBM(ChainableModel):

    def build_model(self, params_dict):
        #self.mode = mode
        #self.params_dict = params_dict
        #self.eval_res = {}
        return lgb.LGBMRegressor(**params_dict)

    #def build_model(self, params_dict):
    def fit_model(self, X, y, X_val, y_val):
        self.model.fit(X, y, eval_set=[(X_val, y_val)], eval_metric='regression_l1', verbose=1,
                           eval_names='validation_set')
        fig, ax = plt.subplots(figsize=(12,10))
        a = lgb.plot_importance(self.model.booster_, ax=ax)
        plt.subplot(a)
        plt.show()

    def validate(self):
    #TODO: DO NOT DELETE IS USEFULL FOR FINISH THE FIT METHOD IN CASE OF VALIDATION

        def _hera_callback(param):
            iteration_num = param[2]
            if iteration_num % param[1]['print_every'] == 0:
                message = f'PARAMS:\n'
                for k in param[1]:
                    message += f'{k}: {param[1][k]}\n'
                Hera.send_message(f'ITERATION_NUM: {iteration_num}\n {message}\n MRR: {param[5][0][2]}', account='edo')

        # define a callback that will insert whitin the dictionary passed the history of the MRR metric during
        # the training phase
        eval_callback = lgb.record_evaluation(self.eval_res)

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




if __name__ == '__main__':
    weather_cols = [f'WEATHER_{i}' for i in range(-10, 0)]
    categorical_cols = ['EMERGENCY_LANE', 'ROAD_TYPE', 'EVENT_DETAIL', 'EVENT_TYPE'] + weather_cols
    X, y = data.dataset_with_features('train', onehot=False)
    X.fillna(0, inplace=True)
    X = reduce_mem_usage(X)


    params_dict = {
        'objective': 'regression_l1',
        'boosting_type':'gbdt',
        'num_leaves': 21,
        'max_depth': 8,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'subsample_for_bin': 200000,
        'class_weights': None,
        'min_split_gain': 0.001,
        'min_child_weight': 0.01,
        'min_child_samples': 1,
        'subsample': 1.0,
        'subsample_freq': 0,
        'colsample_bytree': 0.8,
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
    model_wrapper = RegressorChain(model)
    model_wrapper.fit(X, y)














