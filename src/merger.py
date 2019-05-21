import pandas as pd
from utils.constants import *
from utils.paths import resources_path
from tqdm import tqdm
import os

os.chdir('..')

if __name__ == '__main__':
    speeds_df = pd.read_csv(resources_path('dataset', 'originals', 'speeds_train.csv.gz'))
    events_df = pd.read_csv(resources_path('dataset', 'originals', 'events_train.csv.gz'))

    speeds = []
    for idx, row in tqdm(events_df.iterrows(), total=len(events_df)):
        speed = speeds_df[(speeds_df.KEY == row[KEY]) &
                          (((speeds_df.KM >= row[KM_START]) &
                            (speeds_df.KM <= row[KM_END])) |
                           ((speeds_df.KM <= row[KM_START]) &
                            (speeds_df.KM >= row[KM_END]))) &
                          (speeds_df.DATETIME_UTC >= row[START_DATETIME]) &
                          (speeds_df.DATETIME_UTC <= row[END_DATETIME])]
        speed.drop_duplicates()
        if len(speed) != 0:
            speed[EVENT_TYPE] = row[EVENT_TYPE]
            print(f'found {speed}')
            speeds.append(speed)
            speeds.append(speed)

    speeds = pd.DataFrame(speeds)
    speeds.to_csv(resources_path('dataset', 'merged.csv.gz'))


