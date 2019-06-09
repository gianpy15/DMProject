import src.data as data
from src.algorithms.lightGBM import lightGBM
from src.algorithms.catboost_model import CatBoost
from src.algorithms.multioutput import MultiOutputRegressor
import src.utils.exporter as exporter
from src.algorithms.multioutput import RegressorChain

def to_cat(df):
    weather_cols = [col for col in df.columns if col.startswith('WEATHER_')]
    categorical_cols = ['EMERGENCY_LANE', 'ROAD_TYPE', 'EVENT_DETAIL', 'EVENT_TYPE',
                        'WEEK_DAY', 'IS_WEEKEND'] + weather_cols
    for c in categorical_cols:
        df[c] = df[c].astype('category')
    return df

def seghe_del_catboost(df):
    weather_cols = ['WEATHER_-4', 'WEATHER_-3', 'WEATHER_-2', 'WEATHER_-1']
    df[weather_cols] = df[weather_cols].fillna('Unknown')
    weather_clusters_cols = ['WEATHER_-4_CL', 'WEATHER_-3_CL', 'WEATHER_-2_CL', 'WEATHER_-1_CL']
    df[weather_clusters_cols] = df[weather_clusters_cols].fillna('Unknown')
    return df


if __name__ == '__main__':


    X, y = data.dataset('local', 'train', onehot=False)

    weather_cols = [col for col in X.columns if col.startswith('WEATHER_')]
    categorical_cols = ['EMERGENCY_LANE', 'ROAD_TYPE', 'EVENT_DETAIL', 'EVENT_TYPE'] + weather_cols
    categorical_cols.extend(['WEEK_DAY', 'IS_WEEKEND'])

    X = seghe_del_catboost(X)
    X = to_cat(X)

    X_test, y_test, sub_base_structure = data.dataset('local', 'test', onehot=False, export=True)
    X_test = seghe_del_catboost(X_test)
    X_test = to_cat(X_test)

    params_dict = {
        'objective': 'regression_l1',
        'boosting_type': 'gbdt',
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
#    model = lightGBM(params_dict=params_dict)
#    model_wrapper = MultiOutputRegressor(model)
#   model_wrapper.fit(X, y)
#   predictions_1 = model_wrapper.predict(X_test)
#########################################################à

    chain_mode = input('Choose the chain mode (m: multioutput / c: regressorchain): ').lower()
    M = MultiOutputRegressor if chain_mode == 'm' else RegressorChain



    catboost = CatBoost({
        'X': X,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'n_estimators': 10000,
        'depth': 3,
        'learning_rate': 0.1,
        'early_stopping_rounds': 100,
        'cat_features': categorical_cols
    })

    model = M(catboost)
    model.fit(X, y)
    predictions_2 = model.predict(X_test)


    s_pred1 = exporter.export_sub(sub_base_structure,predictions_1)
    s_pred2 = exporter.export_sub(sub_base_structure, predictions_2)

    dict_scores = {
        'bs': [s_pred1, s_pred2],
        'weights': [0.5, 0.5]
    }

    s_pred_hybrid = exporter.hybrid_score(dict_scores)
    exporter.compute_MAE(s_pred_hybrid, y_test)

