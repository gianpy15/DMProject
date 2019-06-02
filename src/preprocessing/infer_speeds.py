import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
#if you want to know current working dir
sys.path.append(os.getcwd())

from src.utils import *
from src.utility import merge_speed_events
import src.data as data
import src.utility as utils
from src.utils import resources_path
from src.preprocessing.other_features import avg_speed_for_roadtype_event
from tqdm import tqdm


if __name__ == '__main__':
    print('Reading datasets...')
    X_df = data.base_dataset()
    speeds = data.speeds()
    print('Done')
    
    speeds[DATETIME] = pd.to_datetime(speeds[DATETIME])
    print('Inferring...')
    for i in tqdm(range(1, 11)):
        time = 'DATETIME_UTC_-' + str(i)
        speed_avg = 'SPEED_AVG_-' + str(i)
        speed_max = 'SPEED_MAX_-' + str(i)
        speed_min = 'SPEED_MIN_-' + str(i)
        speed_std = 'SPEED_SD_-' + str(i)
        n_cars = 'N_VEHICLES_-' + str(i)
        X_df[time] = pd.to_datetime(X_df[time])

        X_df.drop(columns=[speed_avg, speed_max, speed_min, speed_std, n_cars], inplace=True)
        X_df = pd.merge(X_df, speeds[[KEY, KM, DATETIME, SPEED_AVG, SPEED_MAX, SPEED_MIN, SPEED_SD, N_CARS]],
                        left_on=[KEY, KM, time], right_on=[KEY, KM, DATETIME], how='left')
        X_df.rename(columns={SPEED_AVG: speed_avg, SPEED_MAX: speed_max, SPEED_MIN: speed_min, SPEED_SD: speed_std, N_CARS: n_cars}, inplace=True)
    
    print('Done')
    columns = [c for c in X_df if c.startswith('DATETIME_UTC_x') or c.startswith('DATETIME_UTC_y') or c.startswith('SPEED_AVG_Y')]
    X_df.drop(columns=columns, inplace=True)
    path = resources_path('dataset', 'preprocessed', 'base_dataframe_train_inferred.csv.gz')
    X_df.to_csv(path, compression='gzip')
    print('Saved dataset to ' + path)
    