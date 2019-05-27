from src import data
from src.utils.folder import create_if_does_not_exist

"""
    build a dictionary where:
    key:   station id
    value: dictionary where:
        key:   station_id
        value: estimated distance between the two stations
    
    returns the dictionary
"""
def compute_meteo_station_distances(distances_df):
    splitted = [x.replace(';', ',') for x in distances_df.STATIONS.values]
    d = {}
    for e in splitted:
        stations = e.split(',')[0::2]
        for s in stations:
            if s not in d:
                d[s] = {}
    for e in splitted:
        distances = list(map(float, e.split(',')[1::2]))
        stations = e.split(',')[0::2]
        j = 0
        while j < len(distances) - 1:
            i = j+1
            dist = distances[i] + distances[j]
            
            if stations[j] not in d[stations[i]]:
                d[stations[i]][stations[j]] = dist
            else:
                if d[stations[i]][stations[j]] > dist:
                    d[stations[i]][stations[j]] = dist
            
            if stations[i] not in d[stations[j]]:
                d[stations[j]][stations[i]] = dist
            else:
                if d[stations[j]][stations[i]] > dist:
                    d[stations[j]][stations[i]] = dist

            j += 1
    return d

"""
    given a column name and a dataframe, 
    sorts the stations encoded in form
    station_id,dist;...
    in distance order
"""
def sort_station_by_closest(distances_df, column):
    stations = distances_df[column]
    ordered_stations = []
    for s in stations:
        s_splitted_semicolon_unsorted = s.split(';')
        s_expanded_unsorted = [[c.split(',')[0], float(c.split(',')[1])] for c in s_splitted_semicolon_unsorted]
        s_expanded_sorted = sorted(s_expanded_unsorted, key = lambda x: int(x[1]))
        s_splitted_semicolon_unsorted = ['{},{}'.format(c[0], c[1]) for c in s_expanded_sorted]
        ordered_stations.append(';'.join(s_splitted_semicolon_unsorted))
    distances_df[column] = ordered_stations
    return distances_df

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
        string = ';'.join(['{},{}'.format(split[2*i], split[2*i + 1]) for i in range(int(len(split)/2))])
        stations_splitted_nice.append(string)
    distances_df.STATIONS = stations_splitted_nice

    # infer about other stations: put into distances_df, in each sensor,
    # informations about the distance of also other meteorological stations
    d = compute_meteo_station_distances(distances_df)
    stations = distances_df.STATIONS.values
    inferred_stations = []
    for s in stations:
        inferred_string = ''
        splitted = s.split(';')
        stations_already_present = set()
        for e in splitted:
            station = e.split(',')[0]
            stations_already_present.add(station)
        for e in splitted:
            distanc = e.split(',')[1]
            for item in d[station].items():
                if item[0] not in stations_already_present:
                    inferred_string += '{},{};'.format(item[0], float(item[1]) + float(distanc))
                    stations_already_present.add(item[0])
        inferred_stations.append(inferred_string[:-1])

    # append the inferred stations to the original stations
    appended_stations = []
    stations = distances_df.STATIONS.values
    for idx in range(len(stations)):
        if len(inferred_stations[idx]) > 0:
            appended_stations.append('{};{}'.format(stations[idx], inferred_stations[idx]))
        else:
            appended_stations.append(stations[idx])
    distances_df['STATIONS'] = appended_stations

    distances_df = sort_station_by_closest(distances_df, 'STATIONS')

    # finally save
    distances_df.to_csv(distances_df_path, index=False, compression='gzip')

if __name__ == "__main__":
    preprocess()
