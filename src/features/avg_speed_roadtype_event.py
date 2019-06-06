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
class AvgSpeedRoadTypeEvent(FeatureBase):
    
    def __init__(self, mode):
        name = 'avg_speed_road_type_event'
        super(AvgSpeedRoadTypeEvent, self).__init__(
            name=name, mode=mode)

    def extract_feature(self):
        f = None

        if self.mode == 'local':
            tr = data.speeds_original('train')
            te = data.speed_test_masked()
            f = pd.concat([tr, te])
            del tr
            del te
        
        elif self.mode == 'full':
            tr = data.speeds(mode='full')
            te = data.speeds_original('test2')
            f = pd.concat([tr, te])
            del tr
            del te

        etr = data.events(self.mode, 'train')
        ete = data.events(self.mode, 'test')
        ef = pd.concat([etr, ete])
        del etr
        del ete
        sensors = data.sensors()

        m = pd.merge(ef, f, left_on=['KEY', 'DATETIME_UTC'], right_on=['KEY', 'DATETIME_UTC'])
        m = m[(m.KM >= m.KM_START) & (m.KM <= m.KM_END)]
        msensors = pd.merge(m, sensors)
        return msensors[['EVENT_TYPE', 'ROAD_TYPE', 'SPEED_AVG', 'SPEED_SD', 'SPEED_MIN', 'SPEED_MAX', \
          'N_VEHICLES']].groupby(['EVENT_TYPE', 'ROAD_TYPE']).mean().reset_index()\
              .rename(columns={'SPEED_AVG': 'avg_speed_roadtype_event',
                               'SPEED_SD': 'avg_speed_sd_roadtype_event',
                               'SPEED_MIN': 'avg_speed_min_roadtype_event',
                               'SPEED_MAX': 'avg_speed_max_roadtype_event',
                               'N_VEHICLES': 'avg_n_vehicles_roadtype_event'})


if __name__ == '__main__':
    from src.utils.menu import mode_selection
    mode = mode_selection()
    c = AvgSpeedRoadTypeEvent(mode)

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature())

