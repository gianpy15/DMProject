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


class EventSensorDistance(FeatureBase):
    """ Compute the distance between the event and the sensor """
    
    def __init__(self, mode):
        name = 'event_sensor_distance'
        super(EventSensorDistance, self).__init__(
            name=name)

    def extract_feature(self):

        if self.mode == 'local':
            tr = data.speeds_original('train')
            te = data.speed_test_masked()
            f = pd.concat([tr, te])
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

        m = pd.merge(ef, f, left_on=['KEY', 'DATETIME_UTC'], right_on=['KEY', 'DATETIME_UTC'])
        m = m[(m.KM >= m.KM_START) & (m.KM <= m.KM_END)]

        df['start_event_distance'] = df[]
        return df


if __name__ == '__main__':
    c = AVGSpeedDayBefore()

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature())

