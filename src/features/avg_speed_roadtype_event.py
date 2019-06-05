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


class AvgSpeedRoadTypeEvent(FeatureBase):
    
    def __init__(self):
        name = 'avg_speed_road_type_event'
        super(AvgSpeedRoadTypeEvent, self).__init__(
            name=name)

    def extract_feature(self):
        tr = data.speeds_original('train')
        te = data.speed_test_masked()
        speeds = pd.concat([tr, te])
        del tr
        del te
        
        tr = data.events('train')
        te = data.events('test')
        events = pd.concat([tr, te])
        del tr
        del te

        sensors = data.sensors()
        merged = utility.merge_speed_events(speeds, events)

        merged = pd.merge(merged, sensors, on=[KEY, KM])
        merged = merged[[EVENT_TYPE, SPEED_AVG, ROAD_TYPE]].dropna() \
                .groupby([EVENT_TYPE, ROAD_TYPE]).agg(['mean', 'std'])

        merged['AVG_SPEED_EVENT'] = merged[SPEED_AVG]['mean']
        merged['STD_SPEED_EVENT'] = merged[SPEED_AVG]['std']
        merged.columns = merged.columns.droplevel(level=1)

        merged.drop([SPEED_AVG], axis=1, inplace=True)
        merged.reset_index(inplace=True)
        print(merged.head(2))
        return merged


if __name__ == '__main__':
    c = AvgSpeedRoadTypeEvent()

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature())

