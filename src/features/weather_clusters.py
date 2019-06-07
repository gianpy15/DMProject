import numpy as np
from tqdm import tqdm_notebook as tqdm
import src.data as data
from src.utils import *
import src.utils.folder as folder
from src import utility
from src.utils.datetime_converter import convert_to_datetime
import pandas as pd
from src import data
from src.features.feature_base import FeatureBase
import sys
import os
sys.path.append(os.getcwd())


class Weather_clusters(FeatureBase):

    def __init__(self, mode):
        name = 'weather_clusters'
        super(Weather_clusters, self).__init__(
            name=name, mode=mode, columns_to_onehot=[('WEATHER_CL', 'single')])

    def extract_feature(self):
        bs = data.weather()

        bs = bs.drop(["DISTANCE", "TEMPERATURE", "MAX_TEMPERATURE",
                      "MIN_TEMPERATURE", "KEY", "KM", "DATETIME_UTC"], axis=1)
        bs.loc[bs["WEATHER"] == "Quasi sereno", "WEATHER"] = "Quasi Sereno"
        bs.drop_duplicates(inplace=True)

        # -------------------------------------------------
        # Clusters
        clst_1 = ['Forte Pioggerella',
                  'Temporale con Forte  Neve Pioggia e  Grandine']
        clst_2 = ['Temporale con Debole Neve Pioggia e Grandine',
                  'Temporale con Grandine']
        clst_3 = ['Forte Pioggia', 'Rovescio con Forte Pioggia']
        clst_4 = ['Debole Pioggia e Pioggrella', 'Temporale con Debole Pioggia',
                  'Rovescio con Grandine Piccola', 'Rovescio con Debole Pioggia', 'Debole Pioggia']
        clst_5 = ['Debole Pioggia e Neve', 'Debole Neve']
        clst_6 = ['Tempesta di Polvere', 'Nelle Vicinanze Tempesta di Polvere']
        clst_7 = ['Debole Neve a Granuli', 'Neve e Pioggia']
        clst_8 = ['Nelle Vicinanze Nebbia', 'Banchi di Nebbia',
                  'Sottili Banchi di Nebbia', 'Foschia']

        clusters = [clst_1, clst_2, clst_3,
                    clst_4, clst_5, clst_6, clst_7, clst_8]
        clusters_names = ['Forte Pioggerella', 'Temporale con Debole Neve Pioggia e Grandine', 'Forte Pioggia',
                          'Debole Pioggia e Pioggrella', 'Debole Pioggia e Neve', 'Tempesta di Polvere', 'Debole Neve a Granuli', 'Nelle Vicinanze Nebbia']
        # -------------------------------------------------

        print("Processing Clusters")
        bs["WEATHER_CL"] = ""
        it = 0

        for cl in clusters:
            for c in cl:
                bs.loc[bs["WEATHER"] == c, "WEATHER_CL"] = clusters_names[it]
            it = it+1

        print("Done")
        print("Setting to their actual value weathers outside clusters")
        bs.loc[bs['WEATHER_CL'] == "", "WEATHER_CL"] = bs['WEATHER']
        print("Done")

        return bs

    def join_to(self, df, one_hot=False):
        f = self.read_feature(one_hot=one_hot)
        f.append(pd.DataFrame([[np.nan, np.nan]], columns=[
                 'WEATHER', 'WEATHER_CL']), ignore_index=True)
        matching = [i for i in range(len(df.columns.values)) if "WEATHER_-" in df.columns.values[i]]
        for i in matching:
            f_ = f.rename(columns={'WEATHER': df.columns.values[i], 'WEATHER_CL': '{}_CL'.format(df.columns.values[i])})
            df = pd.merge(df, f_, left_on=[df.columns.values[i]], right_on=df.columns.values[i], how='left')
        return df

if __name__ == '__main__':
    from src.utils.menu import mode_selection
    mode = mode_selection()
    c = Weather_clusters(mode)

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature())
