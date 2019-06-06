import sys
import os
sys.path.append(os.getcwd())

from src.features.feature_base import FeatureBase
from src import data
import pandas as pd


class AvgSpeedSensor(FeatureBase):
    """
    say for each street the avg speed
    | KEY | avg_speed_sensor | avg_speed_sd_sensor | avg_speed_min_sensor | avg_speed_max_sensor | avg_n_vehicles_sensor
    """

    def __init__(self):
        name = 'avg_speed_sensor'
        super(AvgSpeedSensor, self).__init__(
            name=name)

    def extract_feature(self):
        df = None

        if self.mode == 'local':
            tr = data.speeds_original('train')
            te = data.speed_test_masked()
            df = pd.concat([tr, te])
            del tr
            del te
        
        elif self.mode == 'full':
            tr = data.speeds(mode='full')
            te = data.speeds_original('test2')
            df = pd.concat([tr, te])
            del tr
            del te
        
        return df[['KEY', 'KM', 'SPEED_AVG', 'SPEED_SD', 'SPEED_MIN', 'SPEED_MAX', 'N_VEHICLES']].groupby(['KEY', 'KM']).mean().reset_index()\
            .rename(columns={'SPEED_AVG': 'avg_speed_sensor',\
                             'SPEED_SD': 'avg_speed_sd_sensor', \
                             'SPEED_MIN': 'avg_speed_min_sensor', \
                             'SPEED_MAX': 'avg_speed_max_sensor', \
                             'N_VEHICLES': 'avg_n_vehicles_sensor'})

if __name__ == '__main__':
    c = AvgSpeedSensor()

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature(one_hot=True))
