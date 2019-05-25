import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objs as go
from plotly.offline import iplot


def show_speeds(speed_df, road_key, from_datetime='2018-11-15', to_datetime='2018-11-17', limit_sensors=15):
    plt.figure(figsize=(14,8))
    plt.xticks(rotation=70)
    spe = speed_df[speed_df.KEY == road_key].sort_values('DATETIME_UTC')
    spe = spe[(spe.DATETIME_UTC > pd.to_datetime(from_datetime)) & (spe.DATETIME_UTC < pd.to_datetime(to_datetime))]
    
    kilometers = sorted(spe.KM.unique())
    if limit_sensors == -1:
        limit_sensors = len(kilometers)
    for km in kilometers[0:limit_sensors]:
        speed_at_km = spe[spe.KM == km]
        linewidth = speed_at_km.LANES.values[0] / 2
        plt.plot(speed_at_km.DATETIME_UTC.values, speed_at_km.SPEED_AVG.values, label=str(km), linewidth=linewidth)
    
    plt.legend()
    plt.show()


def show_speeds_with_events(speed_df, road_key, from_datetime='2018-11-15', to_datetime='2018-11-17', sensor=0, event_marker='*'):
    spe = speed_df[speed_df.KEY == road_key].sort_values('DATETIME_UTC')
    spe = spe[(spe.DATETIME_UTC >= pd.to_datetime(from_datetime)) & (spe.DATETIME_UTC <= pd.to_datetime(to_datetime))]
    
    sensors_km = sorted(spe.KM.unique())
    sensors_count = len(sensors_km)
    if sensor >= sensors_count:
        sensor = sensors_count -1
    
    #for km in sensors_km[sensor]:
    km = sensors_km[sensor]
    plots = []
    speed_at_km = spe[spe.KM == km]

    number_of_lanes = speed_at_km.LANES.values[0]
    linewidth = number_of_lanes / 2
    # events involving the sensor
    road_events = speed_at_km[speed_at_km['index'].notnull()]
    event_ids = road_events['index'].unique().astype(int)
    number_of_events = len(event_ids)
    
    print('Sensors: {}'.format(sensors_count))
    print('Number of events: {}'.format(number_of_events))
    print('Lanes: {}'.format(number_of_lanes))
    
    plots.append( go.Scatter(x=speed_at_km.DATETIME_UTC, y=speed_at_km.SPEED_AVG, name=str(km),
                             mode='lines', marker={ 'line':{ 'width': linewidth } }) )
    
    if(number_of_events > 0):
        for evt_id in event_ids:
            e = road_events[road_events['index'] == evt_id]
            plots.append( go.Scatter(x=e.DATETIME_UTC, y=e.SPEED_AVG, mode='markers', 
                             name=f'{[evt_id]} {e.EVENT_TYPE.values[0]} ({e.KM.values[0]})') )
    iplot(plots)
    