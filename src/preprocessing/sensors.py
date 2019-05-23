import numpy as np
import pandas as pd
import src.data as data
from src.utility import check_folder

def preprocess_sensors():
    _BASE_PATH = 'resources/dataset/preprocessed'
    check_folder(_BASE_PATH)

    # drop the duplicates rows
    sensors_df = data.sensors()
    print('dropping duplicates...')
    sensors_df = sensors_df.drop_duplicates()
    print(f'saving to {_BASE_PATH}/sensors.csv.gz')
    sensors_df.to_csv(path_or_buf=f'{_BASE_PATH}/sensors.csv.gz', index=False, compression='gzip')

if __name__ == '__main__':
    preprocess_sensors()
