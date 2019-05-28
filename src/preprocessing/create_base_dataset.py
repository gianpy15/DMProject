import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import random
import src.data as data
import src.utility as utility


def create_base_dataset(steps_behind_event, steps_after_event=3, validation_split=0.2):
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
    for mode in ['train','test']:
        # - speeds
        s = data.speeds(mode).merge(sensors, how='left')
        # - events
        e = data.events(mode)
        # - weather
        # ......

        # join dataframes
        joined_df = utility.merge_speed_events(s, e)

        # create the time windows for each event
        joined_df = utility.time_windows_event(joined_df, steps_behind=steps_behind_event, steps_after=steps_after_event)

        event_beginning_step = steps_behind_event+1
        joined_df = joined_df.round(4).groupby('sample_id').agg({
            'KEY':'first',
            'KM':'first',
            'event_index':lambda x: x.values[event_beginning_step],
            'DATETIME_UTC':list,
            'SPEED_AVG':list, #[list, lambda x: x[0:event_beginning_step].dropna().mean()],
            'SPEED_SD':list,
            'SPEED_MAX':list,
            'SPEED_MIN':list,
            'N_VEHICLES':list,
            'EMERGENCY_LANE':'first',
            'LANES':'first',
            'ROAD_TYPE':'first',
            'EVENT_DETAIL':lambda x: x.values[event_beginning_step],
            'EVENT_TYPE':lambda x: x.values[event_beginning_step]
        })
        
        # split the last m measure in different columns
        def split_prediction_fields(row, event_beginning_step):
            return pd.Series((
                    row.DATETIME_UTC[:event_beginning_step], row.DATETIME_UTC[event_beginning_step:], 
                    row.SPEED_AVG[:event_beginning_step],    row.SPEED_AVG[event_beginning_step:],
                    row.SPEED_SD[:event_beginning_step],     row.SPEED_SD[event_beginning_step:],
                    row.SPEED_MAX[:event_beginning_step],    row.SPEED_MAX[event_beginning_step:],
                    row.SPEED_MIN[:event_beginning_step],    row.SPEED_MIN[event_beginning_step:],
                    row.N_VEHICLES[:event_beginning_step],   row.N_VEHICLES[event_beginning_step:]
            ))
        
        joined_df[['DATETIME_UTC','DATETIME_UTC_Y', 'SPEED_AVG','SPEED_AVG_Y', 'SPEED_SD','SPEED_SD_Y',
                    'SPEED_MAX','SPEED_MAX_Y', 'SPEED_MIN','SPEED_MIN_Y',
                    'N_VEHICLES', 'N_VEHICLES_Y']] = joined_df.apply(split_prediction_fields, axis=1, event_beginning_step=event_beginning_step-1)

        for col_name in ['DATETIME_UTC','DATETIME_UTC_Y', 'SPEED_AVG','SPEED_AVG_Y', 'SPEED_SD','SPEED_SD_Y',
                            'SPEED_MAX','SPEED_MAX_Y','SPEED_MIN','SPEED_MIN_Y', 'N_VEHICLES', 'N_VEHICLES_Y']:
            if col_name.endswith('_Y'):
                new_cols = ['{}_{}'.format(col_name, i) for i in range(0, steps_after_event+1)]
            else:
                new_cols = ['{}_{}'.format(col_name, i) for i in range(-steps_behind_event, 0)]
            
            joined_df[new_cols] = pd.DataFrame(joined_df[col_name].values.tolist(), index=joined_df.index)

        joined_df = joined_df.drop(['DATETIME_UTC','SPEED_AVG','SPEED_SD','SPEED_MAX','SPEED_MIN','N_VEHICLES',
                                    'DATETIME_UTC_Y','SPEED_AVG_Y','SPEED_SD_Y','SPEED_MAX_Y','SPEED_MIN_Y','N_VEHICLES_Y'], axis=1)

        if mode == 'train':
            # take random validation rows
            pass
            # random_indices = random.shuffle(joined_df.index)
            # validation_indices = random_indices[0: int(len(random_indices) * validation_split)]
            # train_df = joined_df.drop(validation_indices)
            # valid_df = joined_df.loc[validation_indices]

        # save the base structure
        filename = 'base_dataframe_{}.csv.gz'.format(mode)
        filepath = os.path.join(data._BASE_PATH_PREPROCESSED, filename)
        print('Saving base dataframe to {}'.format(filepath))
        joined_df.to_csv(filepath, index=False, compression='gzip')
        print('Done\n')    
        del joined_df


if __name__ == '__main__':
    create_base_dataset(steps_behind_event=10, steps_after_event=3)
