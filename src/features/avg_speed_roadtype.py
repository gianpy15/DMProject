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


class AvgSpeedRoadType(FeatureBase):
    
    def __init__(self):
        name = 'avg_speed_road_type'
        super(AvgSpeedRoadType, self).__init__(
            name=name)

    def extract_feature(self):
        print('Loading datasets')
        speeds = data.speeds()
        sensors = data.sensors()
        print('Done')

        df = pd.merge(speeds.dropna(), sensors, left_on=[KEY, KM], right_on=[KEY, KM])
        df[DATETIME] = pd.to_datetime(df.DATETIME_UTC)

        return df[['ROAD_TYPE', 'SPEED_AVG']].groupby('ROAD_TYPE').mean().reset_index()


if __name__ == '__main__':
    c = AvgSpeedRoadType()

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature)

