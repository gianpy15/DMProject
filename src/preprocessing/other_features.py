import argparse
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

import src.utility as utility
import src.utils.folder as folder
from src.utils import *
import src.data as data
import src.utility as utility

def avg_speed_for_roadtype() -> pd.DataFrame:
    print('Loading datasets')
    speeds = data.speeds()
    sensors = data.sensors()
    print('Done')
    
    df = pd.merge(speeds.dropna(), sensors, left_on=[KEY, KM], right_on=[KEY, KM])
    df[DATETIME] = pd.to_datetime(df.DATETIME_UTC)
    
    df = df[['ROAD_TYPE', 'SPEED_AVG']].groupby('ROAD_TYPE').mean()
    
    return df
    
def avg_speed_for_sensor() -> pd.DataFrame:
    tr = data.speeds_original('train')
    te = data.speeds_original('test')
    df = pd.concat([tr, te])
    return df[['KEY', 'KM', 'SPEED_AVG']].groupby(['KEY', 'KM']).mean().reset_index()

def avg_speed_for_street() -> pd.DataFrame:
    tr = data.speeds_original('train')
    te = data.speeds_original('test')
    df = pd.concat([tr, te])
    return df[['KEY', 'SPEED_AVG']].groupby(['KEY']).mean().reset_index()
