from src.features.feature_base import FeatureBase
from src import data
import pandas as pd


class AvgSpeedStreet(FeatureBase):
    """
    say for each street the avg speed
    | KEY | avg_speed_street 
    """

    def __init__(self):
        name = 'avg_speed_street'
        super(AvgSpeedStreet, self).__init__(
            name=name)

    def extract_feature(self):
        tr = data.speeds_original('train')
        te = data.speeds_original('test')
        df = pd.concat([tr, te])
        return df[['KEY', 'SPEED_AVG']].groupby(['KEY']).mean().reset_index().rename(columns={'SPEED_AVG': 'avg_speed_street'})

if __name__ == '__main__':
    c = AvgSpeedStreet()

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature(one_hot=True))