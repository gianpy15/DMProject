import pandas as pd
"""
    automatically converts to datetime all the columns of a dataframe which are 
    strings representing timestamps.
"""
def convert_to_datetime(df):
    matching = [i for i in range(len(df.columns.values)) if "DATETIME" in df.columns.values[i]]
    for pos in matching:
        df.iloc[:, pos] = pd.to_datetime(df.iloc[:, pos])
    return df
