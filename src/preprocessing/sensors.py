import os
import sys
sys.path.append(os.getcwd())
from src import data
import numpy as np
import pandas as pd
import src.data as data
from src.utility import check_folder

def preprocess():
    print('Preprocessing sensors...')

    # drop the duplicates rows
    sensors_df = data.sensors_original()
    print('dropping duplicates...')
    sensors_df = sensors_df.drop_duplicates()
    
    path = data.get_path_preprocessed('', '', 'sensors.csv.gz')
    sensors_df.to_csv(path, index=False, compression='gzip')

if __name__ == '__main__':
    preprocess()
