import lightgbm as lgb
from tqdm import tqdm
import src.data as data
tqdm.pandas()
from src.algorithms.multi_output_regressor_wrapper import MultiOutputRegressorWrapper


import datetime
from skopt.space import Real, Integer, Categorical
from src.utility import reduce_mem_usage


class lightGBM():

    def __init__(self, mode, params_dict):
        self.mode = mode
        self.params_dict = params_dict
        self.eval_res = {}
        self.model = None

    def fit(self, X, y, X_val=None, y_val = None):
        _VALIDATION = False

        # initialize the model
        self.model = lgb.LGBMRegressor(**self.params_dict)

        # reduce the mem_usage of the dataframes and set the correct types to the columns
        X = reduce_mem_usage(X)
        y = reduce_mem_usage(y)

        if (X_val is not None) and (y_val is not None):
            # TODO IS NOT COMPLETED
            print('Validation detected...')
            X_val = reduce_mem_usage(X_val)
            y_val = reduce_mem_usage(y_val)
            self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose = True)
        else:
            print('Train whitout validation...')
            self.model.fit(X, y, verbose=True)

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

    def predict(self):
        # TODO
        pass



if __name__ == '__main__':
    params_dict = {
        'objective': 'regression_l1',
        'boosting_type':'gbdt',
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample_for_bin': 200000,
        'class_weights': None,
        'min_split_gain': 0.0,
        'min_child_weight': 0.001,
        'min_child_samples': 20,
        'subsample':1.0,
        'subsample_freq': 0,
        'colsample_bytree':1.0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'random_state': None,
        'n_jobs': -1,
        'silent': False,
        'importance_type': 'split',
        'metric': 'None',
        'print_every': 100,
    }

    X, y = data.base_dataset('train')

    model = lightGBM(mode='train', params_dict=params_dict)
    model_wrapper = MultiOutputRegressorWrapper(model)
    model_wrapper.fit(X, y)






