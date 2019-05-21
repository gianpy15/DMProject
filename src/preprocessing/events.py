import numpy as np
import pandas as pd

import os
import src.utils.folder as folder

def preprocess(km_influence_range=5):
    """ Preprocess the events dataframe for train and test:
    - KM_START and KM_END are set to be ordered so that KM_START is less than KM_END
    - 
    """
    for mode in ['train','test']:
        filename = 'events_{}.csv.gz'
        events_df = pd.read_csv(os.path.join('dataset/originals', filename.format(mode)))

        start_km = np.minimum(events_df['KM_START'].values, events_df['KM_END'].values)
        end_km = np.maximum(events_df['KM_START'].values, events_df['KM_END'].values)
        events_df['KM_START'] = start_km
        events_df['KM_END'] = end_km

        # find the events for which KM_START and KM_END coincide
        mask_km_start_equal_km_end = (events_df['KM_START'] == events_df['KM_END'])

        # create a new column containing the value of KM_START and KM_END, NaN when the 2 values are not equal
        events_df.loc[mask_km_start_equal_km_end, 'KM_EVENT'] = events_df.loc[mask_km_start_equal_km_end, 'KM_START']

        # modifiy KM_START and KM_END to be in the range of 5km
        events_df.loc[mask_km_start_equal_km_end, 'KM_START'] -= km_influence_range
        events_df.loc[mask_km_start_equal_km_end, 'KM_END'] += km_influence_range

        # save the df
        preprocessing_folder = 'dataset/preprocessed'
        path = os.path.join(preprocessing_folder, filename.format(mode))
        folder.create_if_does_not_exist(preprocessing_folder)
        events_df.to_csv(path)


