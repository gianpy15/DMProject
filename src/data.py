import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
import os

# base path to original data
_BASE_PATH_ORIGINALS = 'resources/dataset/originals'
_BASE_PATH_PREPROCESSED = 'resources/dataset/preprocessed'

# initialize variable for caching
_distances_df = None
_sensors_df = None
_sensors_df_preprocessed = None
_events_df = {'train': None, 'test': None}
_speeds_df = {'train': None, 'test': None}
_weather_df = {'train': None, 'test': None}
_base_structure_df = None

def base_structure():
    import src.preprocessing.create_base_structure as create_base_structure
    
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
    print(f'base structure loaded in: {round(time() - start_t, 4)} s\n')
    print('shape of the dataframe is: {}'.format(_base_structure_df.shape))
    return _base_structure_df

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

def speeds_train():
    start_t = time()
    if _speeds_df['train'] is None:
        print('caching speeds_train\n')
        _speeds_df['train'] = pd.read_csv(f'{_BASE_PATH_ORIGINALS}/speeds_train.csv.gz', engine='c')
    print(f'speeds_train loaded in: {round(time()-start_t,4)} s\n')
    print('shape of the dataframe is: {}'.format(_speeds_df['train'].shape))
    return _speeds_df['train']

def speeds_test():
    start_t = time()
    if _speeds_df['test'] is None:
        print('caching speeds_test\n')
        _speeds_df['test'] = pd.read_csv(f'{_BASE_PATH_ORIGINALS}/speeds_test.csv.gz', engine='c')
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



