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
    
    def __init__(self):
        name = 'n_vehicles_perDay'
        super(N_vehicles_perDay, self).__init__(
            name=name)
        
        
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
        tr = data.speeds_original('train')
        te = data.speed_test_masked()
        speeds = pd.concat([tr, te])
        del tr
        del te

        feature_cols = ["DATETIME_UTC", "KEY", "KM", "N_VEHICLES"]
        speeds = speeds.loc[:, feature_cols]
        speeds["N_VEHICLES"]=speeds.N_VEHICLES.astype(int)
        #contains also weekday
        speeds["day"]=speeds.DATETIME_UTC.dt.weekday
        speeds=speeds[['KEY','KM','N_VEHICLES','day']].groupby(['KEY','KM','day']).mean().reset_index()

        return speeds

if __name__ == '__main__':
    c = N_vehicles_perDay()

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature)