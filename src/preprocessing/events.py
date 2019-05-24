import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

import src.utility as utility
import src.utils.folder as folder

def preprocess(km_influence_before=5, km_influence_after=2):
    """ Preprocess the events dataframe for train and test:
    - KM_START and KM_END are set to be ordered so that KM_START is less than KM_END
    - if KM_START is equal to KM_END, a new column is created containing the original value and KM_START
        is decreased by km_influence_before and KM_END is increased by km_influence_after to create a valid
        range
    - expand the timestamps creating new rows for the intermediate timestamps (ex: event from 13:12 to 13:43 will be expanded
        to 3 rows: 13:15, 13:30, 13:45)
    """
    for mode in ['train','test']:
        filename = 'events_{}.csv.gz'
        events_df = pd.read_csv(os.path.join('resources/dataset/originals', filename.format(mode)))

        start_km = np.minimum(events_df['KM_START'].values, events_df['KM_END'].values)
        end_km = np.maximum(events_df['KM_START'].values, events_df['KM_END'].values)
        events_df['KM_START'] = start_km
        events_df['KM_END'] = end_km

        # find the events for which KM_START and KM_END coincide
        mask_km_start_equal_km_end = (events_df['KM_START'] == events_df['KM_END'])

        # create a new column containing the value of KM_START and KM_END, NaN when the 2 values are not equal
        events_df.loc[mask_km_start_equal_km_end, 'KM_EVENT'] = events_df.loc[mask_km_start_equal_km_end, 'KM_START']

        # modifiy KM_START and KM_END to be in the range of 5km
        events_df.loc[mask_km_start_equal_km_end, 'KM_START'] -= km_influence_before
        events_df.loc[mask_km_start_equal_km_end, 'KM_END'] += km_influence_after

        # expand the timestamps
        utility.expand_timestamps(events_df, col_ts_start='START_DATETIME_UTC', col_ts_end='END_DATETIME_UTC')

        # save the df
        preprocessing_folder = 'resources/dataset/preprocessed'
        path = os.path.join(preprocessing_folder, filename.format(mode))
        folder.create_if_does_not_exist(preprocessing_folder)
        events_df.to_csv(path, index=False, compression='gzip')

if __name__ == "__main__":
    preprocess()
