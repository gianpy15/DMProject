import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time

# base path to original data
_BASE_PATH_ORIGINALS = 'resources/dataset/originals'

# initialize variable for caching
_distances_df = None
_sensors_df = None
_events_df = {'train': None, 'test': None}
_speeds_df = {'train': None, 'test': None}
_weather_df = {'train': None, 'test': None}

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

if __name__ == '__main__':
    a = speeds_train()



