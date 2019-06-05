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


class AVGSpeedDayBefore(FeatureBase):
    
    def __init__(self):
        name = 'avg_speed_day_before'
        super(AVGSpeedDayBefore, self).__init__(
            name=name)

    def extract_feature(self):
        tr = data.speeds_original('train')
        te = data.speed_test_masked()
        speeds = pd.concat([tr, te])
        del tr
        del te

        print('Extracting min and max timestamps...')
        min_datetime = speeds.DATETIME_UTC.min()
        max_datetime = speeds.DATETIME_UTC.max()
        sensors = data.sensors().drop_duplicates([KEY, KM])
        print('Done')
        
        datetimes_df = pd.DataFrame(pd.date_range(min_datetime, max_datetime, freq='15min').to_series()).reset_index()
        datetimes_df[DATETIME] = pd.to_datetime(datetimes_df['index'])
        datetimes_df = datetimes_df[[DATETIME]]
        print('Shifting hours')
        datetimes_df['DATETIME_HOUR'] = pd.to_datetime(datetimes_df[DATETIME]).apply(lambda x: x.floor('1H'))
        datetimes_df['DATETIME_HOUR'] = datetimes_df['DATETIME_HOUR'] - pd.DateOffset(1)
        print('Done')
        
        print('Creating skeleton')
        datetimes_df['MERGE'] = 0
        sensors['MERGE'] = 0
        skeleton = pd.merge(sensors[[KEY, KM, 'MERGE']], datetimes_df, on='MERGE')
        skeleton[DATETIME] = pd.to_datetime(skeleton[DATETIME])
        skeleton.set_index(DATETIME, inplace=True)
        print('Done')
        
        print('Merging with speeds..')
        resampled_speeds = speeds\
            .groupby([KEY, KM])\
            .apply(lambda x: x.set_index(DATETIME)\
            .resample('H').mean()[[SPEED_AVG, SPEED_MAX, SPEED_MIN, SPEED_SD, N_CARS]]).reset_index()
        skeleton_merge = skeleton.reset_index()
        df = pd.merge(skeleton_merge,
                      resampled_speeds,
                      left_on=[KEY, KM, 'DATETIME_HOUR'],
                      right_on=[KEY, KM, DATETIME])
        df = df.rename(columns={'DATETIME_UTC_x': 'DATETIME_UTC', SPEED_AVG: 'SPEED_AVG_D-1',
                       SPEED_MAX: 'SPEED_MAX_D-1', SPEED_MIN: 'SPEED_MIN_D-1',
                        SPEED_SD: 'SPEED_SD_D-1', N_CARS: 'N_VEHICLES_D-1'})
        print('Done')
        return df


if __name__ == '__main__':
    c = AVGSpeedDayBefore()

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature())

