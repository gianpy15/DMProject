from src.utils.datetime_converter import convert_to_datetime
import pandas as pd
from src import data
from src.features.feature_base import FeatureBase
import sys
import os
sys.path.append(os.getcwd())


class AvgSpeedSensorHour(FeatureBase):
    """
    say for each street the avg speed
    | KEY | KM | DATETIME_UTC_SPEED_SENSOR_HOUR| avg_speed_sensor_hour | avg_speed_sd_sensor_hour | avg_speed_min_sensor_hour | avg_speed_max_sensor_hour | avg_n_vehicles_sensor_hour
    """

    def __init__(self, mode):
        name = 'avg_speed_sensor_hour'
        super(AvgSpeedSensorHour, self).__init__(
            mode=mode, name=name)

    def extract_feature(self):
        df = None

        if self.mode == 'local':
            tr = data.speeds_original('train')
            te = data.speed_test_masked()
            df = pd.concat([tr, te])
            del tr
            del te
        
        elif self.mode == 'full':
            tr = data.speeds(mode='full')
            te = data.speeds_original('test2')
            df = pd.concat([tr, te])
            del tr
            del te
        
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
        feature_df.DATETIME_UTC_SPEED_SENSOR_HOUR = feature_df.DATETIME_UTC_SPEED_SENSOR_HOUR.dt.strftime(
            '%H:%M:%S')

        feature_df_y_0 = feature_df.rename(columns={'avg_speed_sensor_hour': 'avg_speed_sensor_hour_y_0',
                                                    'avg_speed_sd_sensor_hour': 'avg_speed_sd_sensor_hour_y_0',
                                                    'avg_speed_min_sensor_hour': 'avg_speed_min_sensor_hour_y_0',
                                                    'avg_speed_max_sensor_hour': 'avg_speed_max_sensor_hour_y_0',
                                                    'avg_n_vehicles_sensor_hour': 'avg_n_vehicles_sensor_hour_y_0'})
        df['DATETIME_UTC_y_0_m'] = pd.to_datetime(df.DATETIME_UTC_y_0)
        df['DATETIME_UTC_y_0_m'] = df['DATETIME_UTC_y_0_m'].dt.strftime(
            '%H:%M:%S')
        df = df.merge(feature_df_y_0, left_on=['KEY', 'KM', 'DATETIME_UTC_y_0_m'], right_on=[
                      'KEY', 'KM', 'DATETIME_UTC_SPEED_SENSOR_HOUR'])
        df = df.drop(
            ['DATETIME_UTC_y_0_m', 'DATETIME_UTC_SPEED_SENSOR_HOUR'], axis=1)

        feature_df_y_1 = feature_df.rename(columns={'avg_speed_sensor_hour': 'avg_speed_sensor_hour_y_1',
                                                    'avg_speed_sd_sensor_hour': 'avg_speed_sd_sensor_hour_y_1',
                                                    'avg_speed_min_sensor_hour': 'avg_speed_min_sensor_hour_y_1',
                                                    'avg_speed_max_sensor_hour': 'avg_speed_max_sensor_hour_y_1',
                                                    'avg_n_vehicles_sensor_hour': 'avg_n_vehicles_sensor_hour_y_1'})
        df['DATETIME_UTC_y_1_m'] = pd.to_datetime(df.DATETIME_UTC_y_1)
        df['DATETIME_UTC_y_1_m'] = df['DATETIME_UTC_y_1_m'].dt.strftime(
            '%H:%M:%S')
        df = df.merge(feature_df_y_1, left_on=['KEY', 'KM', 'DATETIME_UTC_y_1_m'], right_on=[
                      'KEY', 'KM', 'DATETIME_UTC_SPEED_SENSOR_HOUR'])
        df = df.drop(
            ['DATETIME_UTC_y_1_m', 'DATETIME_UTC_SPEED_SENSOR_HOUR'], axis=1)

        feature_df_y_2 = feature_df.rename(columns={'avg_speed_sensor_hour': 'avg_speed_sensor_hour_y_2',
                                                    'avg_speed_sd_sensor_hour': 'avg_speed_sd_sensor_hour_y_2',
                                                    'avg_speed_min_sensor_hour': 'avg_speed_min_sensor_hour_y_2',
                                                    'avg_speed_max_sensor_hour': 'avg_speed_max_sensor_hour_y_2',
                                                    'avg_n_vehicles_sensor_hour': 'avg_n_vehicles_sensor_hour_y_2'})
        df['DATETIME_UTC_y_2_m'] = pd.to_datetime(df.DATETIME_UTC_y_2)
        df['DATETIME_UTC_y_2_m'] = df['DATETIME_UTC_y_2_m'].dt.strftime(
            '%H:%M:%S')
        df = df.merge(feature_df_y_2, left_on=['KEY', 'KM', 'DATETIME_UTC_y_2_m'], right_on=[
                      'KEY', 'KM', 'DATETIME_UTC_SPEED_SENSOR_HOUR'])
        df = df.drop(
            ['DATETIME_UTC_y_2_m', 'DATETIME_UTC_SPEED_SENSOR_HOUR'], axis=1)

        feature_df_y_3 = feature_df.rename(columns={'avg_speed_sensor_hour': 'avg_speed_sensor_hour_y_3',
                                                    'avg_speed_sd_sensor_hour': 'avg_speed_sd_sensor_hour_y_3',
                                                    'avg_speed_min_sensor_hour': 'avg_speed_min_sensor_hour_y_3',
                                                    'avg_speed_max_sensor_hour': 'avg_speed_max_sensor_hour_y_3',
                                                    'avg_n_vehicles_sensor_hour': 'avg_n_vehicles_sensor_hour_y_3'})
        df['DATETIME_UTC_y_3_m'] = pd.to_datetime(df.DATETIME_UTC_y_3)
        df['DATETIME_UTC_y_3_m'] = df['DATETIME_UTC_y_3_m'].dt.strftime(
            '%H:%M:%S')
        df = df.merge(feature_df_y_3, left_on=['KEY', 'KM', 'DATETIME_UTC_y_3_m'], right_on=[
                      'KEY', 'KM', 'DATETIME_UTC_SPEED_SENSOR_HOUR'])
        df = df.drop(
            ['DATETIME_UTC_y_3_m', 'DATETIME_UTC_SPEED_SENSOR_HOUR'], axis=1)

        return df


if __name__ == '__main__':
    from src.utils.menu import mode_selection
    mode = mode_selection()
    c = AvgSpeedSensorHour(mode)

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature(one_hot=True))
