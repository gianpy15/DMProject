import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
import os
import pandas as pd
import src.utility as utility

# base path to original data
_BASE_PATH_ORIGINALS = 'resources/dataset/originals'
_BASE_PATH_PREPROCESSED = 'resources/dataset/preprocessed'

# initialize variable for caching
_distances_df_original = None
_distances_df_preprocessed = None
_sensors_df = None
_sensors_df_preprocessed = None
_events_df = {'train': None, 'test': None}
_events_df_preprocessed = {'train': None, 'test': None}

_speeds_df = {'train': None, 'test': None}
_weather_df = {'train': None, 'test': None}
_base_structure_df = None
_base_dataset_df = {'train': None, 'test': None}
_base_structure_hours_df = None


def check_mode(mode):
    assert mode in ['train', 'test']

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


def base_dataset(mode='train'):
    check_mode(mode)
    import src.preprocessing.create_base_dataset as create_base_dataset

    base_dataset_path = os.path.join(_BASE_PATH_PREPROCESSED, f'base_dataframe_{mode}.csv.gz')
    if _base_dataset_df[mode] is None:
        if not os.path.isfile(base_dataset_path):
            print('base dataset not found... creating it...')
            create_base_dataset.create_base_dataset()    
    
        print('caching base dataset {}'.format(mode))
        _base_dataset_df[mode] = pd.read_csv(base_dataset_path)
        _base_dataset_df[mode] = utility.df_to_datetime(_base_dataset_df[mode],
                                        columns=['START_DATETIME_UTC','END_DATETIME_UTC','DATETIME_UTC'])

    return _base_dataset_df[mode]

def events_original(mode='train'):
    check_mode(mode)
    if _events_df[mode] is None:
        filepath = f'{_BASE_PATH_ORIGINALS}/events_{mode}.csv.gz'
        print(f'caching {filepath}')
        _events_df[mode] = pd.read_csv(filepath, engine='c')
    
    return _events_df[mode]

def events(mode='train'):
    check_mode(mode)
    if _events_df_preprocessed[mode] is None:
        filepath = f'{_BASE_PATH_PREPROCESSED}/events_{mode}.csv.gz'
        print(f'caching {filepath}')
        _events_df_preprocessed[mode] = pd.read_csv(filepath, engine='c')
        _events_df_preprocessed[mode] = utility.df_to_datetime(_events_df_preprocessed[mode],
                                                columns=['START_DATETIME_UTC','END_DATETIME_UTC','DATETIME_UTC'])
    
    return _events_df_preprocessed[mode]


def speeds(mode='train'):
    check_mode(mode)
    if _speeds_df[mode] is None:
        filepath = f'{_BASE_PATH_ORIGINALS}/speeds_{mode}.csv.gz'
        print(f'caching {filepath}')
        _speeds_df[mode] = pd.read_csv(filepath, engine='c')
        _speeds_df[mode] = utility.df_to_datetime(_speeds_df[mode], columns=['DATETIME_UTC'])

    return _speeds_df[mode]


def weather_original(mode='train'):
    check_mode(mode)
    if _weather_df[mode] is None:
        filepath = f'{_BASE_PATH_ORIGINALS}/weather_{mode}.csv.gz'
        print(f'caching {filepath}')
        _weather_df[mode] = pd.read_csv(filepath, engine='c')
        _weather_df[mode] = utility.df_to_datetime(_weather_df[mode], columns=['DATETIME_UTC'])
    
    return _weather_df[mode]



def sensors_original():
    global _sensors_df
    start_t = time()
    if _sensors_df is None:
        print('caching sensors\n')
        _sensors_df = pd.read_csv(f'{_BASE_PATH_ORIGINALS}/sensors.csv.gz', engine='c')
    print(f'sensors loaded in: {round(time() - start_t, 4)} s\n')
    print('shape of the dataframe is: {}'.format(_sensors_df.shape))
    return _sensors_df

def sensors():
    global _sensors_df_preprocessed
    start_t = time()
    if _sensors_df_preprocessed is None:
        print('caching sensors\n')
        _sensors_df_preprocessed = pd.read_csv(f'{_BASE_PATH_PREPROCESSED}/sensors.csv.gz', engine='c')
    print(f'sensors loaded in: {round(time() - start_t, 4)} s\n')
    print('shape of the dataframe is: {}'.format(_sensors_df_preprocessed.shape))
    return _sensors_df_preprocessed

def distances_original():
    global _distances_df_original
    start_t = time()
    if _distances_df_original is None:
        print('caching distances\n')
        _distances_df_original = pd.read_csv(f'{_BASE_PATH_ORIGINALS}/distances.csv.gz',sep='|',names = ["KEY_KM", "STATIONS"])
    print(f'distances loaded in: {round(time() - start_t, 4)} s\n')
    print('shape of the dataframe is: {}'.format(_distances_df_original.shape))
    return _distances_df_original

def distances_proprocessed():
    global _distances_df_preprocessed
    start_t = time()
    if _distances_df_preprocessed is None:
        print('caching distances\n')
        _distances_df_preprocessed = pd.read_csv(f'{_BASE_PATH_PREPROCESSED}/distances.csv.gz')
    print(f'distances loaded in: {round(time() - start_t, 4)} s\n')
    print('shape of the dataframe is: {}'.format(_distances_df_preprocessed.shape))
    return _distances_df_preprocessed


