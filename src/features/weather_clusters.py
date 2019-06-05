import sys
import os
sys.path.append(os.getcwd())

from src.features.feature_base import FeatureBase
from src import data
import pandas as pd
from src.utils.datetime_converter import convert_to_datetime
from src import utility
import src.utils.folder as folder
from src.utils import *
import src.data as data
from tqdm import tqdm_notebook as tqdm




class Weather_clusters(FeatureBase):
    
    def __init__(self):
        name = 'weather_clusters'
        super(Weather_clusters, self).__init__(
            name=name)

    def extract_feature(self):
        bs=data.weather()
        
        bs=bs.drop(["DISTANCE","TEMPERATURE","MAX_TEMPERATURE","MIN_TEMPERATURE","KEY","KM","DATETIME_UTC"],axis=1)
        bs.loc[bs["WEATHER"]=="Quasi sereno","WEATHER"]="Quasi Sereno"
        bs.drop_duplicates(inplace=True)
        
        #-------------------------------------------------
        #Clusters 
        clst_1=['Forte Pioggerella','Temporale con Forte  Neve Pioggia e  Grandine']
        clst_2=['Temporale con Debole Neve Pioggia e Grandine','Temporale con Grandine']
        clst_3=['Forte Pioggia','Rovescio con Forte Pioggia']
        clst_4=['Debole Pioggia e Pioggrella','Temporale con Debole Pioggia','Rovescio con Grandine Piccola','Rovescio con Debole Pioggia','Debole Pioggia']
        clst_5=['Debole Pioggia e Neve','Debole Neve']
        clst_6=['Tempesta di Polvere','Nelle Vicinanze Tempesta di Polvere']
        clst_7=['Debole Neve a Granuli','Neve e Pioggia']
        clst_8=['Nelle Vicinanze Nebbia','Banchi di Nebbia','Sottili Banchi di Nebbia','Foschia']

        clusters=[clst_1,clst_2,clst_3,clst_4,clst_5,clst_6,clst_7,clst_8]
        clusters_names=['Forte Pioggerella','Temporale con Debole Neve Pioggia e Grandine','Forte Pioggia','Debole Pioggia e Pioggrella','Debole Pioggia e Neve','Tempesta di Polvere','Debole Neve a Granuli','Nelle Vicinanze Nebbia']
        #-------------------------------------------------


        print("Processing Clusters")
        bs["WEATHER_CL"]=""
        it=0
        
        for cl in clusters:
            for c in cl:
                bs.loc[bs["WEATHER"]==c,"WEATHER_CL"]=clusters_names[it]
            it = it+1
            
        print("Done")
        print("Setting to their actual value weathers outside clusters")
        bs.loc[bs['WEATHER_CL']=="", "WEATHER_CL"] = bs['WEATHER']
        print("Done")

        return bs


if __name__ == '__main__':
    c = Weather_clusters()

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature)

