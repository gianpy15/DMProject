import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import src.data as data
import src.preprocessing.speeds as speeds
import src.preprocessing.events as events
import src.preprocessing.sensors as sensors
import src.preprocessing.distances as distances

import src.preprocessing.create_base_dataset as dataset


if __name__ == "__main__":
    """ Preprocess the required dataframe and create the dataset. """
    print()
    speeds.create_speeds_test_for_unbiased_features(data.speeds_original('test'))
    
    print()
    events.preprocess()
    
    print()
    sensors.preprocess()

    print()
    distances.preprocess()
    
    print()
    dataset.create_base_dataset(steps_behind_event=4, steps_after_event=3)
