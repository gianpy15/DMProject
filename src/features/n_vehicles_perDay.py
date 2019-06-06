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


class N_vehicles_perDay(FeatureBase):
    
    def __init__(self, mode):
        name = 'n_vehicles_perDay'
        super(N_vehicles_perDay, self).__init__(
            name=name, mode=mode)
        
        
    def join_to(self, df, one_hot=False):
        """ Join this feature to the specified dataframe. The default implementation will join based on the
        common column between the 2 dataframes. Override to provide a custom join logic. """
        feature_df = self.read_feature(one_hot=one_hot)
        
        df["DATETIME_UTC"]=df.DATETIME_UTC_y_0
        df["day"]=df.DATETIME_UTC.dt.weekday
        merged = df.merge(feature_df, how='left')
        merged = merged.drop(["day","DATETIME_UTC"],axis=1)
        return merged

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

        feature_cols = ["DATETIME_UTC", "KEY", "KM", "N_VEHICLES"]
        speeds = speeds.loc[:, feature_cols]
        speeds["N_VEHICLES"]=speeds.N_VEHICLES.fillna(0).astype(int)
        #contains also weekday
        speeds["day"]=speeds.DATETIME_UTC.dt.weekday
        speeds=speeds[['KEY','KM','N_VEHICLES','day']].groupby(['KEY','KM','day']).mean().reset_index()

        return speeds.rename(columns={'N_VEHICLES': 'avg_n_vehicles_sensor_per_day'})

if __name__ == '__main__':
    from src.utils.menu import mode_selection
    mode = mode_selection()
    c = N_vehicles_perDay(mode)

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature())