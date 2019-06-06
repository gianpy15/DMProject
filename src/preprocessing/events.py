import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from src import data
import src.utility as utility
import src.utils.folder as folder

def function(events_df, km_influence_before, km_influence_after):
    # reorder KM_START and KM_END, such that KM_START <= KM_END
    start_km = np.minimum(events_df['KM_START'].values, events_df['KM_END'].values)
    end_km = np.maximum(events_df['KM_START'].values, events_df['KM_END'].values)
    events_df['KM_START'] = start_km
    events_df['KM_END'] = end_km

    # find the events for which KM_START and KM_END coincide
    mask_km_start_equal_km_end = (events_df['KM_START'] == events_df['KM_END'])
    # create a new column containing the value of KM_START and KM_END, NaN when the 2 values are not equal
    events_df.loc[mask_km_start_equal_km_end, 'KM_EVENT'] = events_df.loc[mask_km_start_equal_km_end, 'KM_START']

    # modifiy KM_START and KM_END for all the events
    events_df['KM_START'] -= km_influence_before
    events_df['KM_END'] += km_influence_after

    # remove all events that do not involve a time step
    # events_df = utility.discretize_timestamp(events_df, col_name='START_DATETIME_UTC', rename_col='next_datetime_step')
    # events_df = events_df[events_df['END_DATETIME_UTC'] >= events_df['next_datetime_step']]
    # events_df.drop('next_datetime_step', axis=1, inplace=True)

    # expand the timestamps
    events_df = utility.expand_timestamps(events_df, col_ts_start='START_DATETIME_UTC', col_ts_end='END_DATETIME_UTC',
                                            ceil_if_rem0=True)

    events_df['START_DATETIME_UTC'] = pd.to_datetime(events_df['START_DATETIME_UTC'], unit='s')
    events_df['END_DATETIME_UTC'] = pd.to_datetime(events_df['END_DATETIME_UTC'], unit='s')
    events_df['DATETIME_UTC'] = pd.to_datetime(events_df['DATETIME_UTC'], unit='s')

    # put into event detail a unique value when is NaN
    events_df.EVENT_DETAIL = events_df.EVENT_DETAIL.fillna(-1)

    return events_df


def preprocess(mode='local', km_influence_before=2, km_influence_after=2):
    """ Preprocess the events dataframe for train and test:
    - KM_START and KM_END are set to be ordered so that KM_START is less than KM_END
    - if KM_START is equal to KM_END, a new column is created containing the original value
    - ALL events KM_START is decreased by km_influence_before
    - ALL events KM_END is increased by km_influence_after
    - expand the timestamps creating new rows for the intermediate timestamps (ex: event from 13:12 to 13:43 will be expanded
        to 3 rows: 13:15, 13:30, 13:45)
    """
    print('Preprocessing events...')

    if mode == 'local':
        for t in ['train','test']:
            events_df = data.events_original(t)

            events_df = function(events_df, km_influence_before, km_influence_after)

            # save the df
            path = data.get_path_preprocessed(mode, t, 'events.csv.gz')

            print('saving {}'.format(path))
            events_df.to_csv(path, index=False, compression='gzip')

    elif mode == 'full':
        events_local_train = data.events_original('train')
        events_local_test = data.events_original('test')
        events_full = pd.concat([events_local_train, events_local_test]).reset_index(drop=True)
        del events_local_train
        del events_local_test

        events_train_full = function(events_full, km_influence_before, km_influence_after)

        path = data.get_path_preprocessed(mode, 'train', 'events.csv.gz')
        events_train_full.to_csv(path, index=False, compression='gzip')
        del events_train_full

        events_df = data.events_original('test2')
        events_df = function(events_df, km_influence_before, km_influence_after)
        path = data.get_path_preprocessed(mode, 'test', 'events.csv.gz')
        print('saving {}'.format(path))
        events_df.to_csv(path, index=False, compression='gzip')

if __name__ == "__main__":
    preprocess('local')
    preprocess('full')
