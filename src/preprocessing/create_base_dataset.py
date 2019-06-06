import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import random
import src.data as data
import src.utility as utility
import src.utils.menu as menu
import psutil
from tqdm.auto import tqdm

"""
def create_base_dataset_old(mode, steps_behind_event, steps_after_event=3, validation_split=0.2): #, speed_imputed=True):
    # Create the dataframe containing the road measurements for every timestamp and related
    # additional information about sensors, events and weather
    
    print(f'Creating base dataset for {mode.upper()} with timewindows ({steps_behind_event}, {steps_after_event})')

    # load dataframes to be joined
    # - sensors
    sensors = data.sensors()
    weather = data.weather()

    for t in ['train','test']:
        print()
        print('Creating dataset', t.upper())
        # - speeds
        # if speed_imputed:
        #     s = data.speeds(mode).merge(sensors, how='left')
        # else:
        print('Merging speeds and events...')
        e = data.events(mode, t)
        if mode == 'local':
            s = data.speeds_original(t)
        elif mode == 'full':
            s = data.speeds(mode=mode, t=t)
        se = utility.merge_speed_events(s, e)
        
        print('Done')
        print_memory_usage()

        # create the time windows for each event
        print('Creating time windows for events...')
        joined_df = utility.time_windows_event(se, speeds_df=s, steps_behind=steps_behind_event, steps_after=steps_after_event)

        # add other dataframes
        # - events
        events_info = e.drop(['KEY','KEY_2','DATETIME_UTC','step_duration','START_DATETIME_UTC','END_DATETIME_UTC'],axis=1) \
                        .groupby(['index']).first()
        joined_df = joined_df.merge(events_info, how='left', left_on='event_index', right_index=True)
        # - weather
        joined_df = joined_df.merge(weather, how='left')
        # - sensors
        joined_df = joined_df.merge(sensors, how='left')

        print('Aggregating events in samples...')
        joined_df = joined_df.round(4).groupby('sample_id').agg({
            'KEY':'first',
            'KM':'first',
            'event_index':lambda x: x.values[steps_behind_event],
            'DATETIME_UTC':list,
            'SPEED_AVG':list, #[list, lambda x: x[0:event_beginning_step].dropna().mean()],
            'SPEED_SD':list,
            'SPEED_MAX':list,
            'SPEED_MIN':list,
            'N_VEHICLES':list,
            'EMERGENCY_LANE':'first',
            'LANES':'first',
            'ROAD_TYPE':'first',
            'EVENT_DETAIL':lambda x: x.values[steps_behind_event],
            'EVENT_TYPE':lambda x: x.values[steps_behind_event],
            'WEATHER': list,
            'DISTANCE': list,
            'TEMPERATURE': list,
            'MIN_TEMPERATURE': list,
            'MAX_TEMPERATURE': list
        })
        
        # split the last m measures in different columns
        def split_prediction_fields(row, event_beginning_step):
            return pd.Series((
                    row.DATETIME_UTC[:event_beginning_step], row.DATETIME_UTC[event_beginning_step:], 
                    row.SPEED_AVG[:event_beginning_step],    row.SPEED_AVG[event_beginning_step:],
                    row.SPEED_SD[:event_beginning_step],
                    row.SPEED_MAX[:event_beginning_step],
                    row.SPEED_MIN[:event_beginning_step],
                    row.N_VEHICLES[:event_beginning_step],
                    row.WEATHER[:event_beginning_step],
                    row.DISTANCE[:event_beginning_step],
                    row.TEMPERATURE[:event_beginning_step],
                    row.MIN_TEMPERATURE[:event_beginning_step],
                    row.MAX_TEMPERATURE[:event_beginning_step],
            ))
        
        print('Splitting time steps into separate columns...')
        
        columns_to_split = ['DATETIME_UTC','DATETIME_UTC_y',
                            'SPEED_AVG','SPEED_AVG_Y',
                            'SPEED_SD', 'SPEED_MAX', 'SPEED_MIN', 'N_VEHICLES', 'WEATHER', 'DISTANCE',
                            'TEMPERATURE', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE']
        joined_df[columns_to_split] = joined_df.apply(split_prediction_fields, axis=1, event_beginning_step=steps_behind_event)

        for col_name in columns_to_split:
            if col_name.upper().endswith('_Y'):
                new_cols = ['{}_{}'.format(col_name, i) for i in range(0, steps_after_event+1)]
            else:
                new_cols = ['{}_{}'.format(col_name, i) for i in range(-steps_behind_event, 0)]
            
            joined_df[new_cols] = pd.DataFrame(joined_df[col_name].values.tolist(), index=joined_df.index)

        # removed the residual columns of lists
        joined_df = joined_df.drop(columns_to_split, axis=1)

        # drop the rows for which all speeds are NaNs
        print('Dataset shape:', joined_df.shape)
        #print('Dropping not available speeds...')
        #joined_df.dropna(how='all', subset=[f'SPEED_AVG_{i}' for i in range(-steps_behind_event, 0)], inplace=True)
        #print('Dataset shape reduced to:', joined_df.shape)

        # cast to int some columns
        joined_df = joined_df.astype({'EMERGENCY_LANE': 'int', 'LANES': 'int',
                                      'ROAD_TYPE': 'int', 'EVENT_DETAIL': 'int',
                                      'KEY': 'int', 'KM': 'int', 'event_index':'int'})

        
        #if mode == 'train':
            # take random validation rows

            # random_indices = random.shuffle(joined_df.index)
            # validation_indices = random_indices[0: int(len(random_indices) * validation_split)]
            # train_df = joined_df.drop(validation_indices)
            # valid_df = joined_df.loc[validation_indices]
        

        # save the base dataset
        filepath = data.get_path_preprocessed(mode, t, 'base_dataset.csv.gz')

        print('Saving base dataframe to {}'.format(filepath))
        joined_df.to_csv(filepath, index=False, compression='gzip')
        del joined_df
        print('Done')
"""

def add_possible_sensors(events_df):
    sensors = data.sensors()
    res_df = sensors[['KEY', 'KM']].drop_duplicates().sort_values(['KEY','KM']).groupby('KEY').agg(list)
    res_df = res_df.rename(columns={'KM':'ROAD_SENSORS'})
    return events_df.merge(res_df, on='KEY', how='left')

def merge_speed_events(speed_df, events_df):
    tqdm.pandas()
    events_with_sensor_df = add_possible_sensors(events_df)
    #def in_range()
    events_with_sensor_df['sensors'] = events_with_sensor_df.progress_apply( \
        lambda row: [x for x in row.ROAD_SENSORS if row.KM_START <= x <= row.KM_END], axis=1)
    events_with_sensor_df = events_with_sensor_df[events_with_sensor_df['sensors'].str.len() > 0]
    return events_with_sensor_df.drop('ROAD_SENSORS', axis=1)

def create_base_dataset(mode, steps_behind_event, steps_after_event=3, validation_split=0.2):
    """
    Create the dataframe containing the road measurements for every timestamp and related
    additional information about sensors, events and weather
    """
    print(f'Creating base dataset for {mode.upper()} with timewindows ({steps_behind_event}, {steps_after_event})')

    # load dataframes to be joined
    # - sensors
    sensors = data.sensors()
    weather = data.weather()

    for t in ['train','test']:
        print()
        print('Creating dataset', t.upper())
        # - speeds
        # if speed_imputed:
        #     s = data.speeds(mode).merge(sensors, how='left')
        # else:
        print('Merging speeds and events...')
        e = data.events(mode, t)

        if mode == 'local':
            speeds = data.speeds_original(t)
        elif mode == 'full':
            speeds = data.speeds(mode=mode, t=t)
        
        print('Done')
        print_memory_usage()

        # create the time windows for each event
        print('Creating time windows for events...')

        # find the starting time of each event
        ev_agg = e.astype({'KEY':'int'}).groupby('index').agg({
            'step_duration':'first',
            'EVENT_DETAIL':'first',
            'EVENT_TYPE':'first',
            'KM_END':'first',
            'KM_START':'first',
            'KEY':'first',
            'KEY_2':'first',
            'KM_EVENT':'first',
            'START_DATETIME_UTC':'min',
        }).rename(columns={'step_duration':'event_duration'})

        ev_agg['timewind_start'] = ev_agg.START_DATETIME_UTC - pd.to_timedelta(15*steps_behind_event, unit='m')
        ev_agg['timewind_end'] = ev_agg.START_DATETIME_UTC + pd.to_timedelta(15*steps_after_event, unit='m')

        # add speeds info
        ev_agg = merge_speed_events(speeds, ev_agg)

        # expand different sensors
        base_df = pd.DataFrame({col:np.repeat(ev_agg[col], ev_agg['sensors'].str.len()) \
                           for col in ev_agg.columns.drop('sensors')} \
            ).assign(**{'KM': np.concatenate(ev_agg['sensors'].values)})
        # expand timestamps
        base_df = utility.expand_timestamps(base_df, col_ts_start='timewind_start', col_ts_end='timewind_end')\
                    .drop(['timewind_start','timewind_end','step_duration'], axis=1) \
                    .rename(columns={'index':'event_index'}) \
                    .sort_values('event_index')
        base_df['DATETIME_UTC'] = pd.to_datetime(base_df['DATETIME_UTC'], unit='s')

        joined_df = base_df.drop('KEY_2',axis=1).merge(speeds.astype({'KEY':'int'}), how='left', on=['KEY','KM','DATETIME_UTC'])

        # add other dataframes
        # - weather
        joined_df = joined_df.merge(weather, how='left')
        # - sensors
        joined_df = joined_df.merge(sensors, how='left')

        print('Aggregating events in samples...')
        joined_df = joined_df.sort_values(['KEY','KM','DATETIME_UTC']) \
            .groupby(['event_index','KEY','KM'], as_index=False).agg({
            'DATETIME_UTC':list,
            'event_duration':'first',
            'SPEED_AVG':list, #[list, lambda x: x[0:event_beginning_step].dropna().mean()],
            'SPEED_SD':list,
            'SPEED_MAX':list,
            'SPEED_MIN':list,
            'N_VEHICLES':list,
            'EMERGENCY_LANE':'first',
            'LANES':'first',
            'ROAD_TYPE':'first',
            'EVENT_DETAIL':lambda x: x.values[steps_behind_event],
            'EVENT_TYPE':lambda x: x.values[steps_behind_event],
            'WEATHER': list,
            'DISTANCE': list,
            'TEMPERATURE': list,
            'MIN_TEMPERATURE': list,
            'MAX_TEMPERATURE': list
        })
        
        # split the last m measures in different columns
        def split_prediction_fields(row, event_beginning_step):
            return pd.Series((
                    row.DATETIME_UTC[:event_beginning_step], row.DATETIME_UTC[event_beginning_step:], 
                    row.SPEED_AVG[:event_beginning_step],    row.SPEED_AVG[event_beginning_step:],
                    row.SPEED_SD[:event_beginning_step],
                    row.SPEED_MAX[:event_beginning_step],
                    row.SPEED_MIN[:event_beginning_step],
                    row.N_VEHICLES[:event_beginning_step],
                    row.WEATHER[:event_beginning_step],
                    row.DISTANCE[:event_beginning_step],
                    row.TEMPERATURE[:event_beginning_step],
                    row.MIN_TEMPERATURE[:event_beginning_step],
                    row.MAX_TEMPERATURE[:event_beginning_step],
            ))
        
        print('Splitting time steps into separate columns...')
        
        columns_to_split = ['DATETIME_UTC','DATETIME_UTC_y',
                            'SPEED_AVG','SPEED_AVG_Y',
                            'SPEED_SD', 'SPEED_MAX', 'SPEED_MIN', 'N_VEHICLES', 'WEATHER', 'DISTANCE',
                            'TEMPERATURE', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE']
        joined_df[columns_to_split] = joined_df.apply(split_prediction_fields, axis=1, event_beginning_step=steps_behind_event)

        for col_name in columns_to_split:
            if col_name.upper().endswith('_Y'):
                new_cols = ['{}_{}'.format(col_name, i) for i in range(0, steps_after_event+1)]
            else:
                new_cols = ['{}_{}'.format(col_name, i) for i in range(-steps_behind_event, 0)]
            
            joined_df[new_cols] = pd.DataFrame(joined_df[col_name].values.tolist(), index=joined_df.index)

        # removed the residual columns of lists
        joined_df = joined_df.drop(columns_to_split, axis=1)

        # drop the rows for which all speeds are NaNs
        print('Dataset shape:', joined_df.shape)
        #print('Dropping not available speeds...')
        #joined_df.dropna(how='all', subset=[f'SPEED_AVG_{i}' for i in range(-steps_behind_event, 0)], inplace=True)
        #print('Dataset shape reduced to:', joined_df.shape)

        # set to NaN some of the target speeds if the events is shorter than 4 time steps
        joined_df.loc[joined_df['event_duration'] == 3, 'SPEED_AVG_Y_3'] = np.nan
        joined_df.loc[joined_df['event_duration'] == 2, ['SPEED_AVG_Y_2','SPEED_AVG_Y_3']] = np.nan
        joined_df.loc[joined_df['event_duration'] == 1, ['SPEED_AVG_Y_1','SPEED_AVG_Y_2','SPEED_AVG_Y_3']] = np.nan
        joined_df.drop('event_duration', axis=1, inplace=True)

        # cast to int some columns
        joined_df = joined_df.astype({'EMERGENCY_LANE': 'int', 'LANES': 'int',
                                      'ROAD_TYPE': 'int', 'EVENT_DETAIL': 'int',
                                      'KEY': 'int', 'KM': 'int', 'event_index':'int'})

        """
        if mode == 'train':
            # take random validation rows

            # random_indices = random.shuffle(joined_df.index)
            # validation_indices = random_indices[0: int(len(random_indices) * validation_split)]
            # train_df = joined_df.drop(validation_indices)
            # valid_df = joined_df.loc[validation_indices]
        """

        # save the base dataset
        filepath = data.get_path_preprocessed(mode, t, 'base_dataset.csv.gz')

        print('Saving base dataframe to {}'.format(filepath))
        joined_df.to_csv(filepath, index=False, compression='gzip')
        del joined_df
        print('Done')

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f'Current memory usage: {int(process.memory_info().rss / float(2 ** 20))}MB')


if __name__ == '__main__':
    mode = menu.mode_selection()
    create_base_dataset(mode, steps_behind_event=4, steps_after_event=3)
