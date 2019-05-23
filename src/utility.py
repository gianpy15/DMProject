import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def check_folder(path):
    split_folder = os.path.split(path)
    if '.' in split_folder[1]:
        # path is a file
        path = split_folder[0]
    if not os.path.exists(path):
        print(f'{path} folder created')
        os.makedirs(path, exist_ok=True)

def discretize_timestamp(df, col_name, step=15*60, floor=True, rename_col=None):
    """
    Discretize a datetime column of a dataframe.
    df (dataframe):     dataframe
    col_name (str):     name of the datetime column
    step (int):         interval of discretization
    floor (bool):       whether to approximate to the lower or upper value
    rename_col (str):   name of the new column, None to replace the old one
    """
    unix_timestamps = df[col_name].astype('int64') // 10**9 #s
    remainders = unix_timestamps % step

    if floor:
        times_serie = pd.to_datetime(unix_timestamps - remainders + step, unit='s')
    else:
        times_serie = pd.to_datetime(unix_timestamps - remainders, unit='s')
    
    if isinstance(rename_col, str):
        df[rename_col] = times_serie
    else:
        df[col_name] = times_serie
    return df


def expand_timestamps(df, col_ts_start='START_DATETIME_UTC', col_ts_end='END_DATETIME_UTC'):
    """ Expand the timestamp of a dataframe, in order to have all the intermediate timestamps
        between col_ts_start and col_ts_end.
    """
    tqdm.pandas()
    # cast the columns containing string to datetime
    df[col_ts_start] = pd.to_datetime(df[col_ts_start])
    df[col_ts_end] = pd.to_datetime(df[col_ts_end])
    df = df.sort_values(col_ts_start).reset_index()

    # discretize the timestamps
    df = discretize_timestamp(df, col_name=col_ts_start, floor=True)
    df = discretize_timestamp(df, col_name=col_ts_end, floor=True)

    # cast the datatimes to unix timestamps (s)
    df[col_ts_start] = df[col_ts_start].astype('int') // 10**9   #s
    df[col_ts_end] = df.END_DATETIME_UTC.astype('int') // 10**9  #s

    # compute the difference of the 2 timestamps in time steps of 15 minutes
    df['step_duration'] = (df.END_DATETIME_UTC - df[col_ts_start]) // (15*60) +1

    # build the time range from start to end at steps of 900 seconds (15 minutes)
    df['time_range'] = df.progress_apply(lambda x:
                                    np.arange(x[col_ts_start], x[col_ts_end] + 900, 900), axis=1)
    
    # expand the list of timestamps in the time_range
    df = pd.DataFrame({
        col: np.repeat(df[col].values, df['step_duration'])
        for col in df.columns.drop('time_range')}
    ).assign(**{'time_range': np.concatenate(df['time_range'].values)})

    return df