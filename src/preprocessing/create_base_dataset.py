import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import src.data as data
import src.utility as utils


def create_base_dataset():
    """
    Create the dataframe containing the road measurements for every timestamp and related
    additional information about sensors, events and weather
    """
    # check if the folder exsist, otherwise create it
    utils.check_folder(data._BASE_PATH_PREPROCESSED)

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
        joined_df = utils.merge_speed_events(s, e)

        # save the base structure
        filename = 'base_dataframe_{}.csv.gz'.format(mode)
        filepath = os.path.join(data._BASE_PATH_PREPROCESSED, filename)
        print('Saving base dataframe to {}/'.format(filepath))
        joined_df.to_csv(filepath, index=False, compression='gzip')
        print('Done\n')
        del joined_df


if __name__ == '__main__':
    create_base_dataset()
