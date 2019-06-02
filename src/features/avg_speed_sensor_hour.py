from src.features.feature_base import FeatureBase
from src import data
import pandas as pd
from src.utils.datetime_converter import convert_to_datetime


class AvgSpeedSensorHour(FeatureBase):
    """
    say for each street the avg speed
    | KEY | KM | DATETIME_UTC_SPEED_SENSOR_HOUR| avg_speed_sensor_hour | avg_speed_sd_sensor_hour | avg_speed_min_sensor_hour | avg_speed_max_sensor_hour | avg_n_vehicles_sensor_hour
    """

    def __init__(self):
        name = 'avg_speed_sensor_hour'
        super(AvgSpeedSensorHour, self).__init__(
            name=name)

    def extract_feature(self):
        tr = data.speeds_original('train')
        te = data.speeds_original('test')
        df = pd.concat([tr, te])
        df.DATETIME_UTC = df.DATETIME_UTC.dt.strftime('%H:%M:%S')
        return df[['KEY', 'KM', 'DATETIME_UTC', 'SPEED_AVG', 'SPEED_SD', 'SPEED_MIN', 'SPEED_MAX', 'N_VEHICLES']].groupby(['KEY', 'KM', 'DATETIME_UTC']).mean().reset_index()\
            .rename(columns={'DATETIME_UTC': 'DATETIME_UTC_SPEED_SENSOR_HOUR',
                             'SPEED_AVG': 'avg_speed_sensor_hour',
                             'SPEED_SD': 'avg_speed_sd_sensor_hour',
                             'SPEED_MIN': 'avg_speed_min_sensor_hour',
                             'SPEED_MAX': 'avg_speed_max_sensor_hour',
                             'N_VEHICLES': 'avg_n_vehicles_sensor_hour'})

    def join_to(self, df, one_hot=False):
        feature_df = convert_to_datetime(self.read_feature(one_hot=one_hot))
        feature_df.DATETIME_UTC_SPEED_SENSOR_HOUR = feature_df.DATETIME_UTC_SPEED_SENSOR_HOUR.dt.strftime('%H:%M:%S')

        df['DATETIME_UTC_y_0_m'] = pd.to_datetime(df.DATETIME_UTC_y_0)
        df['DATETIME_UTC_y_0_m'] = df['DATETIME_UTC_y_0_m'].dt.strftime('%H:%M:%S')
        df = df.merge(feature_df, left_on=['KEY', 'KM', 'DATETIME_UTC_y_0_m'], right_on=['KEY', 'KM', 'DATETIME_UTC_SPEED_SENSOR_HOUR'])
        df = df.drop[['DATETIME_UTC_y_0_m', 'DATETIME_UTC_SPEED_SENSOR_HOUR']]

        df['DATETIME_UTC_y_1_m'] = pd.to_datetime(df.DATETIME_UTC_y_1)
        df['DATETIME_UTC_y_1_m'] = df['DATETIME_UTC_y_1_m'].dt.strftime('%H:%M:%S')
        df = df.merge(feature_df, left_on=['KEY', 'KM', 'DATETIME_UTC_y_1_m'], right_on=['KEY', 'KM', 'DATETIME_UTC_SPEED_SENSOR_HOUR'])
        df = df.drop[['DATETIME_UTC_y_1_m', 'DATETIME_UTC_SPEED_SENSOR_HOUR']]

        df['DATETIME_UTC_y_2_m'] = pd.to_datetime(df.DATETIME_UTC_y_1)
        df['DATETIME_UTC_y_2_m'] = df['DATETIME_UTC_y_2_m'].dt.strftime('%H:%M:%S')
        df = df.merge(feature_df, left_on=['KEY', 'KM', 'DATETIME_UTC_y_2_m'], right_on=['KEY', 'KM', 'DATETIME_UTC_SPEED_SENSOR_HOUR'])
        df = df.drop[['DATETIME_UTC_y_2_m', 'DATETIME_UTC_SPEED_SENSOR_HOUR']]

        df['DATETIME_UTC_y_3_m'] = pd.to_datetime(df.DATETIME_UTC_y_1)
        df['DATETIME_UTC_y_3_m'] = df['DATETIME_UTC_y_3_m'].dt.strftime('%H:%M:%S')
        df = df.merge(feature_df, left_on=['KEY', 'KM', 'DATETIME_UTC_y_3_m'], right_on=['KEY', 'KM', 'DATETIME_UTC_SPEED_SENSOR_HOUR'])
        df = df.drop[['DATETIME_UTC_y_3_m', 'DATETIME_UTC_SPEED_SENSOR_HOUR']]


if __name__ == '__main__':
    c = AvgSpeedSensorHour()

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature(one_hot=True))
