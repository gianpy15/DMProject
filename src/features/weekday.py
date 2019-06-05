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


class Weekday(FeatureBase):
    
    def __init__(self):
        name = 'weekday'
        super(Weekday, self).__init__(
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
        print('Done')
        df = pd.DataFrame(pd.date_range(min_datetime, max_datetime, freq='15min').to_series()).reset_index()
        df[DATETIME] = pd.to_datetime(df['index'])
        df = df[[DATETIME]]
        df['WEEK_DAY'] = pd.to_datetime(df[DATETIME]).dt.weekday
        df['IS_WEEKEND'] = df.WEEK_DAY.map(lambda x: x in [5, 6])
        return df


if __name__ == '__main__':
    c = Weekday()

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature())

