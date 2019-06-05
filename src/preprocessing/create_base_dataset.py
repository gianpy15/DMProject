import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import random
import src.data as data
import src.utility as utility
import psutil


def create_base_dataset(steps_behind_event, steps_after_event=3, validation_split=0.2): #, speed_imputed=True):
    """
    Create the dataframe containing the road measurements for every timestamp and related
    additional information about sensors, events and weather
    """
    print(f'Creating base dataset with timewindows ({steps_behind_event}, {steps_after_event})')

    # check if the folder exsist, otherwise create it
    utility.check_folder(data._BASE_PATH_PREPROCESSED)

    # load dataframes to be joined
    # - sensors
    sensors = data.sensors()
    weather = data.weather()

    for mode in ['train','test']:
        # - speeds
        # if speed_imputed:
        #     s = data.speeds(mode).merge(sensors, how='left')
        # else:
        print('Merging speeds and events...')
        e = data.events(mode)
        se = utility.merge_speed_events(data.speeds_original(mode), e)
        
        print('Done')
        #data.flush_cache()
        print_memory_usage()

        # create the time windows for each event
        print('Creating time windows for events...')
        joined_df = utility.time_windows_event(se, mode=mode, steps_behind=steps_behind_event, steps_after=steps_after_event)

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
        print('Dropping not available speeds...')
        joined_df.dropna(how='all', subset=[f'SPEED_AVG_{i}' for i in range(-steps_behind_event, 0)], inplace=True)
        print('Dataset shape reduced to:', joined_df.shape)

        # cast to int some columns
        joined_df = joined_df.astype({'EMERGENCY_LANE': 'int', 'LANES': 'int',
                                        'ROAD_TYPE': 'int', 'EVENT_DETAIL': 'float'})

        """
        if mode == 'train':
            # take random validation rows

            # random_indices = random.shuffle(joined_df.index)
            # validation_indices = random_indices[0: int(len(random_indices) * validation_split)]
            # train_df = joined_df.drop(validation_indices)
            # valid_df = joined_df.loc[validation_indices]
        """

        # save the base structure
        filename = 'base_dataframe_{}.csv.gz'.format(mode)
        filepath = os.path.join(data._BASE_PATH_PREPROCESSED, filename)
        print('Saving base dataframe to {}'.format(filepath))
        joined_df.to_csv(filepath, index=False, compression='gzip')
        print('Done\n')    
        del joined_df

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f'Current memory usage: {int(process.memory_info().rss / float(2 ** 20))}MB')


if __name__ == '__main__':
    create_base_dataset(steps_behind_event=4, steps_after_event=3)
