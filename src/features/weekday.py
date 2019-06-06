import sys
import os
sys.path.append(os.getcwd())
from src.utils.datetime_converter import convert_to_datetime
from src.features.feature_base import FeatureBase
from src import data
import pandas as pd
from src.utils.datetime_converter import convert_to_datetime
from src import utility
import src.utils.folder as folder
from src.utils import *
import src.data as data


class Weekday(FeatureBase):
    
    def __init__(self, mode):
        name = 'weekday'
        super(Weekday, self).__init__(
            name=name, mode=mode)

    def extract_feature(self):
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

        print('Extracting min and max timestamps...')
        min_datetime = speeds.DATETIME_UTC.min()
        max_datetime = speeds.DATETIME_UTC.max()
        print('Done')
        df = pd.DataFrame(pd.date_range(min_datetime, max_datetime, freq='15min').to_series()).reset_index()
        df[DATETIME] = pd.to_datetime(df['index'])
        df = df[[DATETIME]]
        df['WEEK_DAY'] = pd.to_datetime(df[DATETIME]).dt.weekday
        df['IS_WEEKEND'] = df.WEEK_DAY.map(lambda x: 1 if x in [5, 6] else 0)
        return df.rename(columns={'DATETIME_UTC': 'DATETIME_UTC_y_0'})

    def join_to(self, df, one_hot=False):
        """ Join this feature to the specified dataframe. The default implementation will join based on the
        common column between the 2 dataframes. Override to provide a custom join logic. """
        feature_df = convert_to_datetime(self.read_feature(one_hot=one_hot))
        return pd.merge(df, feature_df, how='left')

if __name__ == '__main__':
    from src.utils.menu import mode_selection
    mode = mode_selection()
    c = Weekday(mode)

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature())

