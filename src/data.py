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
_distances_df = None
_sensors_df = None
_sensors_df_preprocessed = None
_events_df = {'train': None, 'test': None}
_events_df_preprocessed = {'train': None, 'test': None}

_speeds_df = {'train': None, 'test': None}
_weather_df = {'train': None, 'test': None}
_base_structure_df = None

def base_structure():
    import src.preprocessing.create_base_structure as create_base_structure
    # HARDCODED start index test
    first_test_index = 15681120

    print('Select the mode:\n')
    print('1) TRAIN\n'
          '2) TEST\n'
          '3) FULL\n')
    _MODE = int(input())

    start_t = time()
    global _base_structure_df
    base_structure_path = f'{_BASE_PATH_PREPROCESSED}/base_structure.csv'
    if _base_structure_df is None:
        if not os.path.isfile(base_structure_path):
            print('base structure not found... creating it...')
            create_base_structure.create_base_structure()
        if _base_structure_df is None:
            print('caching base structure\n')
            _base_structure_df = pd.read_csv(base_structure_path)
            _base_structure_df.DATETIME_UTC = pd.to_datetime(_base_structure_df.DATETIME_UTC)
    print(f'base structure loaded in: {round(time() - start_t, 4)} s\n')
    if _MODE == 1:
        print('train base_structure filtered')
        temp = _base_structure_df[:first_test_index-1]
    elif _MODE == 2:
        print('test base_structure filtered')
        temp = _base_structure_df[first_test_index:]
    else:
        temp = _base_structure_df
    print('shape of the dataframe is: {}'.format(temp.shape))
    return temp

def events_original(mode='train'):
    if _events_df[mode] is None:
        filepath = f'{_BASE_PATH_ORIGINALS}/events_{mode}.csv.gz'
        print(f'caching {filepath}')
        _events_df[mode] = pd.read_csv(filepath, engine='c')
    
    return _events_df[mode]

def events(mode='train'):
    if _events_df_preprocessed[mode] is None:
        filepath = f'{_BASE_PATH_PREPROCESSED}/events_{mode}.csv.gz'
        print(f'caching {filepath}')
        _events_df_preprocessed[mode] = pd.read_csv(filepath, engine='c')
        _events_df_preprocessed[mode] = utility.df_to_datetime(_events_df_preprocessed[mode],
                                                columns=['START_DATETIME_UTC','END_DATETIME_UTC','DATETIME_UTC'])
    
    return _events_df_preprocessed[mode]


def speeds(mode='train'):
    if _speeds_df[mode] is None:
        filepath = f'{_BASE_PATH_ORIGINALS}/speeds_{mode}.csv.gz'
        print(f'caching {filepath}')
        _speeds_df[mode] = pd.read_csv(filepath, engine='c')
        _speeds_df[mode] = utility.df_to_datetime(_speeds_df[mode], columns=['DATETIME_UTC'])

    return _speeds_df[mode]


def weather_original(mode='train'):
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


