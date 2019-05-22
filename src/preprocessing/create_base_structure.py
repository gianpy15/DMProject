import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import src.data as data
import src.utility as utils
from time import time

def create_base_structure():
    """
    Call to create the base structure it is a pd Dataframe composed as follow:
    KEY | DATETIME_UTC | KM
    it is usefull to do join with other dataframe
    in it there are all the DATETIME_UTC present both in train and test speeds.csv files
    """
    start = time()

    # define the base path where to save the base_structure
    _BASE_PATH = 'resources/dataset/preprocessed'

    # check if the folder exsist if not create it
    utils.check_folder(_BASE_PATH)

    speeds_train = data.speeds_train()
    speeds_test = data.speeds_test()

    datetime_train = speeds_train.DATETIME_UTC.unique()
    datetime_test = speeds_test.DATETIME_UTC.unique()

    # get all the unique datetimes of train and test
    datetime_full = set(datetime_train)|set(datetime_test)

    key_2_train = speeds_train.KEY_2.unique()
    key_2_test = speeds_test.KEY_2.unique()

    # get all the unique key_2 in train and test
    key_2_full = sorted(set(key_2_test)|set(key_2_train))

    temp = pd.DataFrame(list(map(lambda x: x.split('_'), key_2_full)), columns=['KEY', 'KM'])
    datetime_df = pd.DataFrame(np.array(list(datetime_full)), columns=['DATETIME_UTC'])

    # add dummy column to let a merge do a cartesian product
    temp['dummy'] = 0
    datetime_df['dummy'] = 0

    print('Doing cartesian product... it will take a while!')
    base_structure = pd.merge(datetime_df, temp).drop(['dummy'], axis=1)
    print('Done\n')

    # reorder the columns in KEY, DATETIME_UTC, KM
    km_series = base_structure.pop('KM')
    datetime_series = base_structure.pop('DATETIME_UTC')

    base_structure['DATETIME_UTC'] = datetime_series
    base_structure['KM'] = km_series

    print('sorting values...')
    base_structure = base_structure.sort_values(['KEY', 'DATETIME_UTC', 'KM']).reset_index(drop=True)
    print('Done\n')

    # save the base structure
    print('Saving base structure to {}/base_structure.csv'.format(_BASE_PATH))
    base_structure.to_csv(f'{_BASE_PATH}/base_structure.csv', index=False)
    print('Done\n')

    print(f'PROCEDURE ENDED SUCCESSFULLY IN: {round(time()-start,4)} s')


if __name__ == '__main__':
    create_base_structure()
