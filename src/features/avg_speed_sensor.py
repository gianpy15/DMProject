from src.features.feature_base import FeatureBase
from src import data
import pandas as pd


class AvgSpeedSensor(FeatureBase):
    """
    say for each street the avg speed
    | KEY | avg_speed_sensor
    """

    def __init__(self):
        name = 'avg_speed_sensor'
        super(AvgSpeedSensor, self).__init__(
            name=name)

    def extract_feature(self):
        tr = data.speeds_original('train')
        te = data.speeds_original('test')
        df = pd.concat([tr, te])
        return df[['KEY', 'KM', 'SPEED_AVG']].groupby(['KEY', 'KM']).mean().reset_index().rename(columns={'SPEED_AVG': 'avg_speed_sensor'})

if __name__ == '__main__':
    c = AvgSpeedSensor()

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature(one_hot=True))
