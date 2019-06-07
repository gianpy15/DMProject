import argparse
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

import src.data as data
import src.utility as utility
from shutil import copyfile
import src.utils.folder as folder
from src.utils import *


def setup_parser():
    _parser = argparse.ArgumentParser(description='Train the selected model')
    _parser.add_argument('-s', '--size', type=int, action='store', default=3)
    _parser.add_argument('-a', '--algorithm', type=str, action='store', choices=['time'], default='time')
    _parser.add_argument('-d', '--data', type=str, action='store', choices=['train', 'test', 'all', '2019'], default='train')
    return _parser


def preprocess(infer_size: int = 3, algorithm: str = 'time', data: str = 'train'):
    print('Preprocessing speeds...')
    speeds_df = {}
    dsets = ['train', 'test', '2019'] if data == 'all' else [data]
    print('Reading datasets')
    if data in ['all', 'train']:
        speeds_df['train'] = pd.read_csv(resources_path('dataset', 'originals', 'speeds_train.csv.gz'))

    if data in ['all', 'test']:
        speeds_df['test'] = pd.read_csv(resources_path('dataset', 'originals', 'speeds_test.csv.gz'))
        
    if data in ['all', '2019']:
        speeds_df['2019'] = pd.read_csv(resources_path('dataset', 'originals', 'speeds_2019.csv.gz'))

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
        complete_df.fillna(method='ffill', inplace=True)
        complete_df.fillna(method='bfill', inplace=True)
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
    print('Creating speeds test without target speeds...')
    e = data.events(mode='local', t='test')
    joined_df = utility.merge_speed_events(speeds_test, e)

    speeds_target = utility.time_windows_event(joined_df, t='test', steps_behind=0, steps_after=3)
    speeds_target.dropna(subset=['KEY'], inplace=True)
    # build a dataframe containing the target speeds, so that it can be joined
    # to the original speeds and reveal the target speeds rows
    filter_target = speeds_target[['KEY','DATETIME_UTC','KM']].drop_duplicates()
    del speeds_target
    # add a dummy column to the filter to perform join
    filter_target['istarget'] = 1

    # join the speeds and the filter
    speeds_filtered = speeds_test.merge(filter_target, how='left')
    # now the target speeds have 1 in the 'istarget' column
    # filter the speeds where the row is not a target
    speeds_filtered = speeds_filtered[speeds_filtered['istarget'].isnull()]
    speeds_filtered.drop('istarget', axis=1, inplace=True)
    
    # save
    path = data.get_path_preprocessed('local', 'test', 'speeds_test_masked.csv.gz')
    speeds_filtered.to_csv(path, compression='gzip')

def create_speeds_train_full():
    print('Saving full train speeds...')
    speeds_train_local = data.speeds_original('train')
    speeds_test_original = data.speeds_original('test')
    speeds_train_full = pd.concat([speeds_train_local, speeds_test_original]).reset_index(drop=True)
    path = data.get_path_preprocessed('full', 'train', 'speeds.csv.gz')
    speeds_train_full.to_csv(path, compression='gzip', index=False)
    print()

def create_speeds_full_test():
    print('Saving full test speeds...')
    source_path = data.get_path_originals('speeds_2019.csv.gz')
    dest_path = data.get_path_preprocessed('full', 'test', 'speeds.csv.gz')
    copyfile(source_path, dest_path)
    print()
    


if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args(sys.argv[1:])
    preprocess(args.size, args.algorithm, args.data)

    # preprocess speeds test
    # create_speeds_test_for_unbiased_features(data.speeds_original('test'))

    # create_speeds_train_full()
    # create_speeds_full_test()
