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

    # finally save
    distances_df.to_csv(distances_df_path, index=False, compression='gzip')

if __name__ == "__main__":
    preprocess()