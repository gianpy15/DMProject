from src.utils.paths import resources_path

import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
import os
import pandas as pd
import src.utility as utility
import gc
from src.utils.datetime_converter import convert_to_datetime

# initialize variable for caching
cache = {
    # original files
    'originals': {
        'train': {
            'events': None,
            'speeds': None,
            'weather': None,
        },
        'test': {
            'events': None,
            'speeds': None,
            'weather': None,
        },
        'test2': {
            'events': None,
            'speeds': None,
            'weather': None,
        },

        'sensors': None,
        'distances': None,  
    },
    # preprocessed files
    'preprocessed': {
        'local': {
            'train': {
                'events': None,
                'speeds': None,
                'base_dataset': None,
                'dataset': None,
            },
            'test': {
                'events': None,
                'speeds': None,
                'speeds_masked': None,
                'base_dataset': None,
                'dataset': None,
            },
        },
        'full': {
            'train': {
                'events': None,
                'speeds': None,
                'base_dataset': None,
                'dataset': None,
            },
            'test': {
                'events': None,
                'speeds': None,
                'base_dataset': None,
                'dataset': None,
            },
        },

        'weather': None,
        'sensors': None,
        'distances': None,
    }
}

# _distances_df_original = None
# _distances_df_preprocessed = None
# _sensors_df = None
# _sensors_df_preprocessed = None

# _events_df_preprocessed = {'train': None, 'test': None}

# _speeds_df = {'train': None, 'test': None}
# _speeds_df_imputed = {'train': None, 'test': None}
# _speed_test_masked = None

# _base_structure_df = None
# _base_dataset_df = {'train': None, 'test': None}
# _merged_dataset_df = {'train': None, 'test': None}
# _base_structure_hours_df = None


# BASE paths and path retrieval
_BASE_PATH = 'resources/dataset/'
def get_path_originals(filename):
    filepath = resources_path(_BASE_PATH, 'originals', filename)
    print(f'caching {filepath}')
    return filepath
def get_path_preprocessed(mode, t, filename):
    filepath = resources_path(_BASE_PATH, 'preprocessed', mode, t, filename)
    print(f'caching {filepath}')
    return filepath
# ========

# EVENTS
def events_original(t='train'):
    check_t(t)
    if cache['originals'][t]['events'] is None:
        filepath = get_path_originals('events_{t}.csv.gz')
        cache['originals'][t]['events'] = convert_to_datetime(pd.read_csv(filepath, engine='c'))

    return cache['originals'][t]['events']

def events(mode='local', t='train'):
    check_mode_and_t(mode, t)
    if cache['preprocessed'][mode][t]['events'] is None:
        filename = 'events_2019.csv' if t == 'test2' else 'events_{t}.csv.gz'
        filepath = get_path_preprocessed(filename)
        cache['preprocessed'][mode][t]['events'] = convert_to_datetime(pd.read_csv(filepath, engine='c'))

    return cache['preprocessed'][mode][t]['events']
# ========

# SPEEDS
def speeds_original(t='train'):
    check_t(t)
    if cache['originals'][t]['speeds'] is None:
        filename = 'speeds_2019.csv' if t == 'test2' else 'speeds_{t}.csv.gz'
        filepath = get_path_originals(filename)
        cache['originals'][t]['speeds'] = convert_to_datetime(pd.read_csv(filepath, engine='c'))

    return cache['originals'][t]['speeds']

def speeds(mode='local', t='train'):
    check_mode_and_t(mode, t)
    if cache['preprocessed'][mode][t]['speeds'] is None:
        filepath = get_path_preprocessed(mode, t, 'speeds_{t}.csv.gz')
        cache['preprocessed'][mode][t]['speeds'] = convert_to_datetime(pd.read_csv(filepath, engine='c'))

    return cache['preprocessed'][mode][t]['speeds']
# ========

# WEATHER
def weather_original(t='train'):
    check_t(t)
    if cache['originals'][t]['weather'] is None:
        filename = 'weather_2019.csv' if t == 'test2' else 'weather_{t}.csv.gz'
        filepath = get_path_originals(filename)
        cache['originals'][t]['weather'] = convert_to_datetime(pd.read_csv(filepath, engine='c'))

    return cache['originals'][t]['weather']

def weather():
    if cache['preprocessed']['weather'] is None:
        filepath = get_path_preprocessed('', '', 'weather_{t}.csv.gz')
        cache['preprocessed']['weather'] = convert_to_datetime(pd.read_csv(filepath, engine='c'))

    return cache['preprocessed']['weather']
# ========


# SENSORS
def sensors_original():
    if cache['originals']['sensors'] is None:
        filepath = get_path_originals('sensors.csv.gz')
        cache['originals']['sensors'] = convert_to_datetime(pd.read_csv(filepath, engine='c'))

    return cache['originals']['sensors']

def sensors():
    if cache['preprocessed']['sensors'] is None:
        filepath = get_path_preprocessed('', '', 'sensors.csv.gz')
        cache['preprocessed']['sensors'] = convert_to_datetime(pd.read_csv(filepath, engine='c'))

    return cache['preprocessed']['sensors']
# ========

# DISTANCES
def distances_original():
    if cache['originals']['distances'] is None:
        filepath = get_path_originals('distances.csv.gz')
        cache['originals']['distances'] = convert_to_datetime(pd.read_csv(filepath, engine='c', sep='|', names=['KEY_KM','STATIONS']))

    return cache['originals']['distances']

def distances():
    if cache['preprocessed']['distances'] is None:
        filepath = get_path_preprocessed('', '', 'distances.csv.gz')
        cache['preprocessed']['distances'] = convert_to_datetime(pd.read_csv(filepath, engine='c'))

    return cache['preprocessed']['distances']
# ========

# DATASET
def base_dataset(mode='local', t='train'):
    check_mode_and_t(mode, t)
    if cache['preprocessed'][mode][t]['base_dataset'] is None:
        filepath = get_path_preprocessed(mode, t, 'base_dataset.csv.gz')
        cache['preprocessed'][mode][t]['base_dataset'] = convert_to_datetime(pd.read_csv(filepath, engine='c'))

    return cache['preprocessed'][mode][t]['base_dataset']

def dataset(mode='local', t='train', onehot=True, drop_index_columns=True):
    check_mode_and_t(mode, t)
    if cache['preprocessed'][mode][t]['dataset'] is None:
        filepath = get_path_preprocessed('merged_dataset.csv.gz')
        cache['preprocessed'][mode][t]['dataset'] = convert_to_datetime(pd.read_csv(filepath, engine='c'))
        
        # SORT BY TIMESTAMP (to replicate their split)
        cache['preprocessed'][mode][t]['dataset'].sort_values('DATETIME_UTC_y_0', inplace=True)

    return split_dataset_X_ycache['preprocessed'][mode][t]['dataset'], onehot, drop_index_columns)
# ========



"""
def speed_test_masked():
    global _speed_test_masked
    if _speed_test_masked is None:
        _speed_test_masked = convert_to_datetime(pd.read_csv(f'{_BASE_PATH_PREPROCESSED}/speeds_test_masked.csv.gz', engine='c', 
                                            index_col=0, parse_dates=True))
    return _speed_test_masked



def base_structure(mode='train'):
    assert mode in ['train', 'test', 'full']
    import src.preprocessing.create_base_structure as create_base_structure
    # HARDCODED start index test
    first_test_index = 15681120

    start_t = time()
    global _base_structure_df
    base_structure_path = f'{_BASE_PATH_PREPROCESSED}/base_structure.csv'
    if _base_structure_df is None:
        if not os.path.isfile(base_structure_path):
            print('base structure not found, creating it...')
            create_base_structure.create_base_structure()
        if _base_structure_df is None:
            print('caching base structure...')
            _base_structure_df = pd.read_csv(base_structure_path)
            _base_structure_df.DATETIME_UTC = pd.to_datetime(_base_structure_df.DATETIME_UTC)
    print(f'base structure loaded in: {round(time() - start_t, 4)} s\n')
    if mode == 'train':
        temp = _base_structure_df[:first_test_index]
    elif mode == 'test':
        temp = _base_structure_df[first_test_index:]
    else:
        temp = _base_structure_df
    return temp


def base_structure_hours():
    import src.preprocessing.create_base_structure as create_base_structure
    start_t = time()
    global _base_structure_hours_df
    base_structure_path = f'{_BASE_PATH_PREPROCESSED}/base_structure_hours.csv'
    if _base_structure_hours_df is None:
        if not os.path.isfile(base_structure_path):
            print('base structure not found... creating it...')
            create_base_structure.create_base_structure_hours()
        if _base_structure_hours_df is None:
            print('caching base structure\n')
            _base_structure_hours_df = pd.read_csv(base_structure_path)
            _base_structure_hours_df.DATETIME_UTC = pd.to_datetime(_base_structure_hours_df.DATETIME_UTC)
    print(f'base structure loaded in: {round(time() - start_t, 4)} s\n')
    print('shape of the dataframe is: {}'.format(_base_structure_hours_df.shape))
    return _base_structure_hours_df

"""



def split_dataset_X_y(dataset, onehot, drop_index_columns):
    # retrieve the target values and move them on Y
    Y_columns = ['SPEED_AVG_Y_0', 'SPEED_AVG_Y_1', 'SPEED_AVG_Y_2', 'SPEED_AVG_Y_3']
    y = dataset[Y_columns]

    TO_DROP = Y_columns
    if drop_index_columns:
        TO_DROP.extend(['KEY', 'KM', 'event_index'])

    df = dataset.drop(TO_DROP, axis=1)

    # find the columns where is present DATETIME and filter them
    X = df.filter(regex='^((?!DATETIME).)*$')

    if onehot:
        print('performing onehot')
        columns_to_onehot = []
        for col in X.columns:
            print(col)
            col_type = df[col].dtype
            if col_type == object:
                columns_to_onehot.append(col)
        X = pd.get_dummies(X, columns=columns_to_onehot)

    return X, y



# UTILITY
"""
def flush_cache():
    print('flushing data cache')
    global _distances_df_original,_distances_df_preprocessed,_sensors_df,_sensors_df_preprocessed,_events_df,\
    _events_df_preprocessed, _speeds_df, _speeds_df_imputed, _weather_df, _base_structure_df,_base_dataset_df,\
    _base_structure_hours_df

    del _distances_df_original
    del _distances_df_preprocessed
    del _sensors_df
    del _sensors_df_preprocessed
    del _events_df['train']
    del _events_df['test']
    del _events_df
    del _events_df_preprocessed['train']
    del _events_df_preprocessed['test']
    del _events_df_preprocessed
    
    del _speeds_df['train']
    del _speeds_df['test']
    del _speeds_df
    del _speeds_df_imputed['train']
    del _speeds_df_imputed['test']
    del _speeds_df_imputed
    del _weather_df
    del _base_structure_df
    del _base_dataset_df['train']
    del _base_dataset_df['test']
    del _base_dataset_df
    del _merged_dataset_df['train']
    del _merged_dataset_df['test']
    del _base_structure_hours_df
    gc.collect()
    
    # initialize variable for caching
    _distances_df_original = None
    _distances_df_preprocessed = None
    _sensors_df = None
    _sensors_df_preprocessed = None
    _events_df = {'train': None, 'test': None}
    _events_df_preprocessed = {'train': None, 'test': None}

    _speeds_df = {'train': None, 'test': None}
    _speeds_df_imputed = {'train': None, 'test': None}
    _weather_df = None
    _base_structure_df = None
    _base_dataset_df = {'train': None, 'test': None}
    _base_structure_hours_df = None
"""

def check_mode(mode):
    assert mode in ['local', 'full'], 'Invalid mode!'
def check_t(t):
    assert t in ['train', 'test', 'test2'], 'Invalid type'
def check_mode_and_t(mode, t):
    check_mode(mode)
    check_t(t)
