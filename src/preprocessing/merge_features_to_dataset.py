from src.features.avg_speed_street import AvgSpeedStreet
from src.features.avg_speed_sensor import AvgSpeedSensor
from src.features.avg_speed_sensor_hour import AvgSpeedSensorHour
from src import data

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
            base_dataset = f[0]().join_to(base_dataset, one_hot=f[1])
        else:
            base_dataset = f().join_to(base_dataset, one_hot=default_one_hot)
        print(f'df_shape: {base_dataset.shape}')
    return base_dataset

def merge(features_array, default_one_hot=False):
    save_path = 'resources/dataset/preprocessed/'
    train_base = data.base_dataset('train')
    test_base = data.base_dataset('test')
    merged_train = merge_single_mode(train_base, features_array, default_one_hot)
    print('train completed \n')
    merged_test = merge_single_mode(test_base, features_array, default_one_hot)
    merged_train.to_csv(save_path + 'merged_dataframe_train.csv.gz', compression='gzip', index=False)
    merged_test.to_csv(save_path + 'merged_dataframe_test.csv.gz', compression='gzip', index=False)

if __name__ == '__main__':
    features_array = [AvgSpeedStreet, AvgSpeedSensor, AvgSpeedSensorHour]
    merge(features_array)
