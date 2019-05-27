import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
import os
import pandas as pd

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
_base_structure_hours_df = None

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

def events_train():
    start_t = time()
    if _events_df['train'] is None:
        print('caching events_train\n')
        _events_df['train'] = pd.read_csv(f'{_BASE_PATH_ORIGINALS}/events_train.csv.gz', engine='c')
    print(f'events_train loaded in: {round(time()-start_t,4)} s\n')
    print('shape of the dataframe is: {}'.format(_events_df['train'].shape))
    return _events_df['train']

def events_test():
    start_t = time()
    if _events_df['test'] is None:
        print('caching events_test\n')
        _events_df['test'] = pd.read_csv(f'{_BASE_PATH_ORIGINALS}/events_test.csv.gz', engine='c')
    print(f'events_test loaded in: {round(time()-start_t,4)} s\n')
    print('shape of the dataframe is: {}'.format(_events_df['test'].shape))
    return _events_df['test']

def events_train_preprocessed():
    start_t = time()
    if _events_df_preprocessed['train'] is None:
        print('caching events_train\n')
        _events_df_preprocessed['train'] = pd.read_csv(f'{_BASE_PATH_PREPROCESSED}/events_train.csv.gz', engine='c')
    print(f'events_train loaded in: {round(time()-start_t,4)} s\n')
    print('shape of the dataframe is: {}'.format(_events_df['train'].shape))
    return _events_df_preprocessed['train']

def events_test_preprocessed():
    start_t = time()
    if _events_df_preprocessed['test'] is None:
        print('caching events_test\n')
        _events_df_preprocessed['test'] = pd.read_csv(f'{_BASE_PATH_PREPROCESSED}/events_test.csv.gz', engine='c')
    print(f'events_test loaded in: {round(time()-start_t,4)} s\n')
    print('shape of the dataframe is: {}'.format(_events_df_preprocessed['test'].shape))
    return _events_df_preprocessed['test']

def speeds_train():
    start_t = time()
    if _speeds_df['train'] is None:
        print('caching speeds_train\n')
        _speeds_df['train'] = pd.read_csv(f'{_BASE_PATH_ORIGINALS}/speeds_train.csv.gz', engine='c')
        _speeds_df['train'].DATETIME_UTC = pd.to_datetime(_speeds_df['train'].DATETIME_UTC)
    print(f'speeds_train loaded in: {round(time()-start_t,4)} s\n')
    print('shape of the dataframe is: {}'.format(_speeds_df['train'].shape))
    return _speeds_df['train']

def speeds_test():
    start_t = time()
    if _speeds_df['test'] is None:
        print('caching speeds_test\n')
        _speeds_df['test'] = pd.read_csv(f'{_BASE_PATH_ORIGINALS}/speeds_test.csv.gz', engine='c')
        _speeds_df['test'].DATETIME_UTC = pd.to_datetime(_speeds_df['test'].DATETIME_UTC)
    print(f'speeds_test loaded in: {round(time()-start_t,4)} s\n')
    print('shape of the dataframe is: {}'.format(_speeds_df['test'].shape))
    return _speeds_df['test']

def weather_train():
    start_t = time()
    if _weather_df['train'] is None:
        print('caching weather_train\n')
        _weather_df['train'] = pd.read_csv(f'{_BASE_PATH_ORIGINALS}/weather_train.csv.gz', engine='c')
    print(f'weather_train loaded in: {round(time()-start_t,4)} s\n')
    print('shape of the dataframe is: {}'.format(_weather_df['train'].shape))
    return _weather_df['train']

def weather_test():
    start_t = time()
    if _weather_df['test'] is None:
        print('caching weather_test\n')
        _weather_df['test'] = pd.read_csv(f'{_BASE_PATH_ORIGINALS}/weather_test.csv.gz', engine='c')
    print(f'weather_test loaded in: {round(time()-start_t,4)} s\n')
    print('shape of the dataframe is: {}'.format(_weather_df['test'].shape))
    return _weather_df['test']

def sensors():
    global _sensors_df
    start_t = time()
    if _sensors_df is None:
        print('caching sensors\n')
        _sensors_df = pd.read_csv(f'{_BASE_PATH_ORIGINALS}/sensors.csv.gz', engine='c')
    print(f'sensors loaded in: {round(time() - start_t, 4)} s\n')
    print('shape of the dataframe is: {}'.format(_sensors_df.shape))
    return _sensors_df

def sensors_preprocessed():
    global _sensors_df_preprocessed
    start_t = time()
    if _sensors_df_preprocessed is None:
        print('caching sensors\n')
        _sensors_df_preprocessed = pd.read_csv(f'{_BASE_PATH_PREPROCESSED}/sensors.csv.gz', engine='c')
    print(f'sensors loaded in: {round(time() - start_t, 4)} s\n')
    print('shape of the dataframe is: {}'.format(_sensors_df_preprocessed.shape))
    return _sensors_df_preprocessed

if __name__ == '__main__':
    a = sensors_preprocessed()



