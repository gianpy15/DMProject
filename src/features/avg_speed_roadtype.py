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
# OK

class AvgSpeedRoadType(FeatureBase):
    
    def __init__(self, mode):
        name = 'avg_speed_road_type'
        super(AvgSpeedRoadType, self).__init__(
            name=name, mode=mode)

    def extract_feature(self):
        print('Loading datasets')
        speeds = None

        if self.mode == 'local':
            tr = data.speeds_original('train')
            te = data.speed_test_masked()
            speeds = pd.concat([tr, te])
            del tr
            del te
        
        elif self.mode == 'full':
            tr = data.speeds(mode='full')
            te = data.speeds_original('test2')
            speeds = pd.concat([tr, te])
            del tr
            del te

        sensors = data.sensors()
        print('Done')

        df = pd.merge(speeds.dropna(), sensors, left_on=[KEY, KM], right_on=[KEY, KM])
        df[DATETIME] = pd.to_datetime(df.DATETIME_UTC)

        return df[['ROAD_TYPE', 'SPEED_AVG']].groupby('ROAD_TYPE').mean().reset_index()\
            .rename(columns={'SPEED_AVG': 'avg_speed_roadtype'})

if __name__ == '__main__':
    from src.utils.menu import mode_selection
    mode = mode_selection()
    c = AvgSpeedRoadType(mode)

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature())

