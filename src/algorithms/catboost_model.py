import sys
import os
sys.path.append(os.getcwd())

from src.algorithms.chainable_model import ChainableModel
from src.algorithms.multioutput import MultiOutputRegressor, RegressorChain
from catboost import CatBoostRegressor

from skopt.space import Real, Integer, Categorical
from src.Optimizer import OptimizerWrapper

import src.data as data
import numpy as np
import src.utils.folder as folder
import src.algorithms.inout as inout

from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor

import src.utils.telegram_bot as Melissa

_best_MAE = 100

class CatBoost(ChainableModel):

    def build_model(self, params_dict):
        self.name = 'catboost'
        return CatBoostRegressor(**params_dict)

    def fit_model(self, X, y, X_val, y_val):
        self.model.fit(X, y, eval_set=[(X_val, y_val)])

    def predict(self, X):
        preds = self.model.predict(X)
        return preds

    @staticmethod
    def get_optimize_params():
        space = [
            Real(0.1, 1, name='learning_rate'),
            Integer(2, 8, name='depth'),
            Real(0, 5, name='l2_leaf_reg'),
            #Integer(16, 48, name='max_leaves'),
            Real(0, 2, name='random_strength'),
        ]

        def get_MAE(arg_list):
            keys = ['learning_rate', 'depth', 'l2_leaf_reg', 'random_strength']
            val_params = { keys[i]:arg_list[i] for i in range(len(keys)) }
            #learning_rate, depth, l2_leaf_reg, num_leaves, random_strength = arg_list

            """
            Melissa.send_message(f'starting val CATBOOST\n so fermo nmezzo alla strada... ovviamente\n'
                                 f'{val_params}')
            """
            
            X, Y = data.dataset('local','train', onehot=False)
            
            weather_cols = ['WEATHER_-4','WEATHER_-3','WEATHER_-2','WEATHER_-1']
            X[weather_cols] = X[weather_cols].fillna('Unknown')

            weather_cols = [col for col in X.columns if col.startswith('WEATHER_')]
            categorical_cols = ['EMERGENCY_LANE', 'ROAD_TYPE', 'EVENT_DETAIL','EVENT_TYPE'] + weather_cols
            
            categorical_cols.extend(['WEEK_DAY','IS_WEEKEND'])

            weather_clusters_cols = ['WEATHER_-4_CL','WEATHER_-3_CL','WEATHER_-2_CL','WEATHER_-1_CL']
            X[weather_clusters_cols] = X[weather_clusters_cols].fillna('Unknown')

            # build params from default and validation ones
            params = {
                'X': X,
                'mode': 'local',
                'n_estimators':100000,
                'loss_function': 'MAE',
                'eval_metric': 'MAE',
                
                'early_stopping_rounds': 100,
                'cat_features': categorical_cols
            }
            params.update(val_params)

            catboost = CatBoost(params)
            model = MultiOutputRegressor(catboost)
            model.fit(X, Y)

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
            MAE = inout.evaluate(model, X_test, y_test)

            global _best_MAE
            if MAE<_best_MAE:
                _best_MAE = MAE
                Melissa.send_message(f'CATBOOST\n MAE: {MAE}\nparams:{val_params}\n')
            return MAE

        return space, get_MAE


if __name__ == "__main__":
    import src.utils.menu as menu

    def train_model():
        print()
        mode = menu.mode_selection()
        chain_mode = input('Choose the chain mode (m: multioutput / c: regressorchain): ').lower()
        M = MultiOutputRegressor if chain_mode == 'm' else RegressorChain

        #X, Y = data.dataset_with_features('train', onehot=False, drop_index_columns=True)
        X, Y = data.dataset('local','train', onehot=False)
        print(X.shape, Y.shape)

        # mask_not_all_null = np.any(X[['SPEED_AVG_-4','SPEED_AVG_-3','SPEED_AVG_-2','SPEED_AVG_-1']].notnull(),axis=1)
        # X = X[mask_not_all_null]
        # Y = Y[mask_not_all_null]

        # print('\nAfter cleaning nan')
        # print(X.shape, Y.shape)
        
        weather_cols = ['WEATHER_-4','WEATHER_-3','WEATHER_-2','WEATHER_-1']
        X[weather_cols] = X[weather_cols].fillna('Unknown')

        weather_cols = [col for col in X.columns if col.startswith('WEATHER_')]
        categorical_cols = ['EMERGENCY_LANE', 'ROAD_TYPE', 'EVENT_DETAIL','EVENT_TYPE'] + weather_cols
        
        categorical_cols.extend(['WEEK_DAY','IS_WEEKEND'])

        weather_clusters_cols = ['WEATHER_-4_CL','WEATHER_-3_CL','WEATHER_-2_CL','WEATHER_-1_CL']
        X[weather_clusters_cols] = X[weather_clusters_cols].fillna('Unknown')

        catboost = CatBoost({
            'X': X,
            'mode': mode,
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'n_estimators':5000,
            'depth':6,
            'learning_rate':0.1,
            'early_stopping_rounds': 100,
            'cat_features': categorical_cols
        })
        
        model = M(catboost)
        model.fit(X, Y)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
        mae, mae_4 = inout.evaluate(model, X_test, y_test, intermediate=True)
        print()
        print(mae)
        print(mae_4)

        # save the model
        mae = round(mae, 5)

        suffix = input('Insert model name suffix: ')
        model_folder = 'saved_models'
        folder.create_if_does_not_exist(model_folder)

        chain_mode = 'chain' if chain_mode == 'c' else 'multiout'
        filename = f'catboost_{chain_mode}_{mae}_{suffix}.jl'
        inout.save(model, os.path.join(model_folder, filename))
    
    def validate_model():
        opt = OptimizerWrapper(CatBoost)
        opt.optimize_random()
        
        
    menu.single_choice('What to do?', ['Train','Validate'], [train_model, validate_model])
