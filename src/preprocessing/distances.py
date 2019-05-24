from src import data
from src.utils.folder import create_if_does_not_exist

def preprocess():
    distances_df = data.distances_original()
    distances_df_path = '{}/distances.csv.gz'.format(data._BASE_PATH_PREPROCESSED)
    create_if_does_not_exist(distances_df_path)

    # creo due colonne separate per KEY_KM che per ora sono insieme
    for i,row in distances_df.iterrows():
        tmp=row.KEY_KM
        tmp=tmp.split(',')
        key= tmp[0]
        km=tmp[1]
        distances_df.at[i,'KEY'] = key
        distances_df.at[i,'KM'] = km
    distances_df=distances_df.drop(["KEY_KM"],axis=1)

    # drop nans
    distances_df = distances_df.dropna()

    # add ';' every 2 ',': split is easier !
    stations = distances_df.STATIONS
    stations_splitted_nice = []
    for s in stations:
        split = s.split(',')
        stations_splitted_nice.append(';'.join(['{},{}'.format(split[2*i], split[2*i + 1]) for i in range(int(len(split)/2))]))
    distances_df.STATIONS = stations_splitted_nice

    # order stations by the closest
    stations = distances_df.STATIONS
    ordered_stations = []
    for s in stations:
        s_splitted_semicolon_unsorted = s.split(';')
        s_expanded_unsorted = [[c.split(',')[0], float(c.split(',')[1])] for c in s_splitted_semicolon_unsorted]
        s_expanded_sorted = sorted(s_expanded_unsorted, key = lambda x: int(x[1]))
        s_splitted_semicolon_unsorted = ['{},{}'.format(c[0], c[1]) for c in s_expanded_sorted]
        ordered_stations.append(';'.join(s_splitted_semicolon_unsorted))
    distances_df.STATIONS = ordered_stations

    # finally save
    distances_df.to_csv(distances_df_path, index=False, compression='gzip')

if __name__ == "__main__":
    preprocess()
