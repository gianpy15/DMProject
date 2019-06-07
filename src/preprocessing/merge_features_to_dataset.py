import sys
import os
sys.path.append(os.getcwd())
from src import data
from src.features.avg_speed_street import AvgSpeedStreet
from src.features.avg_speed_sensor import AvgSpeedSensor
from src.features.avg_speed_sensor_hour import AvgSpeedSensorHour
from src.features.speeds_sensor_days_before import SpeedsSensorDaysBefore
from src.features.avg_speed_roadtype import AvgSpeedRoadType
from src.features.avg_speed_roadtype_event import AvgSpeedRoadTypeEvent
from src.features.weather_clusters import Weather_clusters

"""
    merge objects of class feature_base to the base dataset

    features array: array of features class name ([F1, F2, ..]) -> to all of them will be
                        applied the default one hot
                    array of tuples of class names and boolean ([(F1, True), (F2, False), ..]) ->
                        will be applied the onehot only if specified by the boolean attribute
"""
def merge_single_mode(base_dataset, features_array, default_one_hot=False):
    print(f'df_shape: {base_dataset.shape}')
    for f in features_array:
        if type(f) == tuple:
            base_dataset = f.join_to(base_dataset, one_hot=f[1])
        else:
            base_dataset = f.join_to(base_dataset, one_hot=default_one_hot)
        print(f'df_shape: {base_dataset.shape}')
    return base_dataset

def merge_and_return(features_array, mode, default_one_hot=False):
    train_base = data.base_dataset(mode, 'train').copy()
    test_base = data.base_dataset(mode, 'test').copy()
    
    # instantiate the features
    for j in range(len(features_array)):
        features_array[j] = features_array[j](mode)
    
    merged_train = merge_single_mode(train_base, features_array, default_one_hot)
    print('train completed \n')
    merged_test = merge_single_mode(test_base, features_array, default_one_hot)

    return merged_train, merged_test

def merge_and_save(mode, default_one_hot=False):
    features_array = [
        # AvgSpeedStreet,
        # AvgSpeedSensor,
        # AvgSpeedSensorHour,
        # AvgSpeedRoadType,
        # AvgSpeedRoadTypeEvent,
        # SpeedsSensorDaysBefore,
        # Weather_clusters,
    ]
    merged_train, merged_test = merge_and_return(features_array, mode, default_one_hot)

    path_train = data.get_path_preprocessed(mode, 'train', 'merged_dataset.csv.gz')
    merged_train.to_csv(path_train, compression='gzip', index=False)

    path_test = data.get_path_preprocessed(mode, 'test', 'merged_dataset.csv.gz')
    merged_test.to_csv(path_test, compression='gzip', index=False)

if __name__ == '__main__':
    from src.utils.menu import mode_selection
    mode = mode_selection()
    merge_and_save(mode)
