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
    # check if the folder exsist, otherwise create it
    utility.check_folder(data._BASE_PATH_PREPROCESSED)

    # load dataframes to be joined
    # - base structure
    #base = data.base_structure(mode)
    # - sensors
    sensors = data.sensors()
    weather = data.weather()
    for mode in ['train','test']:
        # - speeds
        # if speed_imputed:
        #     s = data.speeds(mode).merge(sensors, how='left')
        # else:
        s = data.speeds_original(mode).merge(sensors, how='left')
        # - events
        e = data.events(mode)
        # - weather
        # ......
        print('Done')
        data.flush_cache()
        print_memory_usage()
        
        # - events
        s = s.merge(weather, how='left')
        e = data.events(mode)

        # join dataframes
        print('Merging speeds and events...')
        joined_df = utility.merge_speed_events(s, e)

        # create the time windows for each event
        print('Creating time windows for events...')
        joined_df = utility.time_windows_event(joined_df, steps_behind=steps_behind_event, steps_after=steps_after_event)

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
            ))
        
        print('Splitting time steps into separate columns...')
        
        columns_to_split = ['DATETIME_UTC','DATETIME_UTC_y',
                            'SPEED_AVG','SPEED_AVG_Y',
                            'SPEED_SD', 'SPEED_MAX', 'SPEED_MIN', 'N_VEHICLES', 'WEATHER', 'DISTANCE']
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
        joined_df.dropna(how='all', subset=[col for col in joined_df.columns if col.startswith('SPEED_AVG_')], inplace=True)
        print('Dataset shape reduced to:', joined_df.shape)

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
    create_base_dataset(steps_behind_event=10, steps_after_event=3)
