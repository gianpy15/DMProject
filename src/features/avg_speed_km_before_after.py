import sys
import os
sys.path.append(os.getcwd())
from src.features.feature_base import FeatureBase
from src import data
import pandas as pd
from src.utils.datetime_converter import convert_to_datetime
from src import utility
import src.utils.folder as folder
from src.utils import *
import src.data as data
import numpy as np

# OK
class AvgSpeedKmBeforeAfter(FeatureBase):
    
    def __init__(self, mode):
        name = 'avg_speed_km_before_after'
        super(AvgSpeedKmBeforeAfter, self).__init__(
            name=name, mode=mode)

    def extract_feature(self):
        print('Reading data...')
        df = data.base_dataset()
        sensors = data.sensors().drop_duplicates().sort_values([KEY, KM])
        speeds = pd.concat([data.speeds_original('train'), data.speeds_original('test'), data.speeds_original('test2')]).drop_duplicates()
        
        sensors['KM_BEFORE'] = sensors['KM'].shift(1)
        sensors['KEY_BEFORE'] = sensors['KEY'].shift(1)
        sensors['KM_AFTER'] = sensors['KM'].shift(-1)
        sensors['KEY_AFTER'] = sensors['KEY'].shift(-1)

        sensors.loc[sensors.KEY_AFTER != sensors.KEY, 'KM_AFTER'] = np.nan
        sensors.loc[sensors.KEY_BEFORE != sensors.KEY, 'KM_BEFORE'] = np.nan

        sensors.drop(['KEY_BEFORE', 'KEY_AFTER'], axis=1, inplace=True)
        sensors = sensors[[KEY, KM, 'KM_BEFORE', 'KM_AFTER']]
        
        merged = pd.merge(df, sensors, left_on=[KEY, KM], right_on=[KEY, KM])
        
        print('creating features...')
        for i in range(1, 5):
            speed_avg_before = 'SPEED_AVG_BEFORE_-' + str(i)
            speed_avg_after = 'SPEED_AVG_AFTER_-' + str(i)
            datetime = 'DATETIME_UTC_-' + str(i)


            speeds[speed_avg_before] = speeds[SPEED_AVG]
            speeds[speed_avg_after] = speeds[SPEED_AVG]
            merged = pd.merge(merged, speeds[[KEY, KM, DATETIME, speed_avg_before]], left_on=[KEY, 'KM_BEFORE', datetime], right_on=[KEY, KM, DATETIME], suffixes=('_x_-' + str(i), '_y_-' + str(i)))

            merged = pd.merge(merged, speeds[[KEY, KM, DATETIME, speed_avg_after]], left_on=[KEY, 'KM_AFTER', datetime], right_on=[KEY, KM, DATETIME], suffixes=('_x_-' + str(i), '_y_-' + str(i)))


        merged.drop(columns=['KM', 'DATETIME_UTC_y_-3', 'KM_y_-3', 'DATETIME_UTC_y_-4',
                             'DATETIME_UTC_y_-2', 'KM_y_-2', 'DATETIME_UTC_y_-1', 'KM_x_-2',
                             'KM_y_-1', 'KM_x_-3',
                             'KM_x_-4', 'KM_y_-4', 'DATETIME_UTC_y_-4'], inplace=True)
        merged.rename(columns={'KM_x_-1': 'KM',
                               'DATETIME_UTC_x_-4': 'DATETIME_UTC_-4',
                               'DATETIME_UTC_x_-3': 'DATETIME_UTC_-3',
                               'DATETIME_UTC_x_-2': 'DATETIME_UTC_-2',
                               'DATETIME_UTC_x_-1': 'DATETIME_UTC_-1'}, inplace=True)
        merged['DELTA_BEFORE'] = merged[KM] - merged['KM_BEFORE']
        merged['DELTA_AFTER'] = merged['KM_AFTER'] - merged[KM]
        
        to_keep_1 = ['DATETIME_UTC_-' + str(k) for k in range(1, 5)]
        to_keep_2 = ['SPEED_AVG_BEFORE_-' + str(k) for k in range(1, 5)]
        to_keep_3 = ['SPEED_AVG_AFTER_-' + str(k) for k in range(1, 5)]
        to_keep_4 = ['DELTA_BEFORE', 'DELTA_AFTER']
        to_keep = [KEY, KM, *to_keep_1, *to_keep_2, *to_keep_3, *to_keep_4]
        
        return merged[to_keep]

    def join_to(self, df, one_hot=False):
        f = convert_to_datetime(self.read_feature())
        return pd.merge(df, f, how='left')

if __name__ == '__main__':
    from src.utils.menu import mode_selection
    mode = mode_selection()
    c = AvgSpeedKmBeforeAfter(mode)

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature())


