import sys
import os
sys.path.append(os.getcwd())

from src.features.feature_base import FeatureBase
from src import data
import pandas as pd

class AvgSpeedStreet(FeatureBase):
    """
    say for each street the avg quantities
    | KEY | avg_speed_street | avg_speed_sd_street | avg_speed_min_street | avg_speed_max_street | avg_n_vehicles_street
    """

    def __init__(self):
        name = 'avg_speed_street'
        super(AvgSpeedStreet, self).__init__(
            name=name)

    def extract_feature(self):
        tr = data.speeds_original('train')
        te = data.speed_test_masked()
        df = pd.concat([tr, te])
        f = df[['KEY', 'SPEED_AVG', 'SPEED_SD', 'SPEED_MIN', 'SPEED_MAX', 'N_VEHICLES']].groupby(['KEY']).mean().reset_index()\
                .rename(columns={'SPEED_AVG': 'avg_speed_street',\
                                'SPEED_SD': 'avg_speed_sd_street', \
                                'SPEED_MIN': 'avg_speed_min_street', \
                                'SPEED_MAX': 'avg_speed_max_street', \
                                'N_VEHICLES': 'avg_n_vehicles_street'})
        return f

if __name__ == '__main__':
    c = AvgSpeedStreet()

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature(one_hot=True))