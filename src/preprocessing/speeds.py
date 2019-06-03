import argparse
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

import src.data as data
import src.utility as utility
import src.utils.folder as folder
from src.utils import *


def setup_parser():
    _parser = argparse.ArgumentParser(description='Train the selected model')
    _parser.add_argument('-s', '--size', type=int, action='store', default=3)
    _parser.add_argument('-a', '--algorithm', type=str, action='store', choices=['time'], default='time')
    _parser.add_argument('-d', '--data', type=str, action='store', choices=['train', 'test', 'all'], default='train')
    return _parser


def preprocess(infer_size: int = 3, algorithm: str = 'time', data: str = 'train'):
    print('Preprocessing speeds...')
    speeds_df = {}
    dsets = ['train', 'test'] if data == 'all' else [data]
    print('Reading datasets')
    if data in ['all', 'train']:
        speeds_df['train'] = pd.read_csv(resources_path('dataset', 'originals', 'speeds_train.csv.gz'))

    if data in ['all', 'test']:
        speeds_df['test'] = pd.read_csv(resources_path('dataset', 'originals', 'speeds_test.csv.gz'))

    sensors_df = pd.read_csv(resources_path('dataset', 'originals', 'sensors.csv.gz'))
    print('Done')
    
    for s in dsets:
        print(f'Fitting values for {s} set')
        min_time = pd.to_datetime(speeds_df[s].DATETIME_UTC).astype('int').min()
        max_time = pd.to_datetime(speeds_df[s].DATETIME_UTC).astype('int').max()
        min_time = min_time // (10 ** 9)
        max_time = max_time // (10 ** 9)
        datetimes = np.arange(min_time, max_time, 15 * 60)
        datetimes = datetimes * (10 ** 9)
        print(f'total datetimes: {len(datetimes)}')

        datetimes = pd.to_datetime(datetimes)
        datetimes = pd.DataFrame({DATETIME: datetimes})

        datetimes['key'] = 0
        sensors = sensors_df[[KEY, KM]].drop_duplicates()
        print(f"total sensors: {len(sensors)}, i'm expecting max {len(datetimes) * len(sensors)} samples")
        sensors['key'] = 0
        skeleton_train_df = pd.merge(sensors, datetimes, on='key')[[KEY, KM, DATETIME]].sort_values([KEY, KM, DATETIME])
        speeds = speeds_df[s]
        speeds[DATETIME] = pd.to_datetime(speeds[DATETIME])
        skeleton = skeleton_train_df[[KEY, KM, DATETIME]]
        print('Merging dataset')
        complete_df = pd.merge(skeleton, speeds, on=[KEY, KM, DATETIME], how='left')[
            [KEY, KM, DATETIME, SPEED_AVG, SPEED_SD, SPEED_MIN, SPEED_MAX, N_CARS]]
        print('Done')
        print(f'Imputing missing values with algorithm {algorithm}')
        complete_df['IMPUTED'] = complete_df.isnull().any(axis=1)
        complete_df[complete_df['IMPUTED'] == True].sort_values([DATETIME]).head(2)

        complete_df[DATETIME] = pd.to_datetime(complete_df[DATETIME])
        complete_df = complete_df.set_index([DATETIME])

        complete_df = complete_df.interpolate(method=algorithm, limit=infer_size, limit_area='inside')
        print('Done')
        complete_df.dropna(subset=[SPEED_AVG], inplace=True)
        print(f'final DataFrame shape: {complete_df.shape}')
        print('Saving CSV')
        complete_df.to_csv(resources_path('dataset', 'preprocessed', 'speeds_{}_imputed_time.csv.gz'.format(s)), compression='gzip')
        print('Done')

        if s == 'test':
            create_speeds_test_for_unbiased_features(complete_df)


def create_speeds_test_for_unbiased_features(speeds_test):
    """ Mask the speeds in test that cannot be used to compute features (i.e.:
    the speeds during the 4 steps to predict).
    Save a new dataframe without those speed measures.
    """
    e = data.events('test')
    joined_df = utility.merge_speed_events(speeds_test, e)

    speeds_target = utility.time_windows_event(joined_df, steps_behind=0, steps_after=3)
    speeds_target.dropna(subset=['KEY'], inplace=True)

    # filter the speeds where the speed is not a target
    speeds_not_target = speeds_target.drop(speeds_target.index)
    # save
    speeds_not_target.to_csv('resources/dataset/preprocessed/speeds_test_masked.csv.gz', compression='gzip')

if __name__ == '__main__':
    # parser = setup_parser()
    # args = parser.parse_args(sys.argv[1:])
    # preprocess(args.size, args.algorithm, args.data)

    # preprocess speeds test
    create_speeds_test_for_unbiased_features(data.speeds_original('test'))