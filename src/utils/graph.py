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


def show_speeds_with_events(speed_df, road_key, from_datetime='2018-11-15', to_datetime='2018-11-17', event_marker='*'):
    spe = speed_df[speed_df.KEY == road_key].sort_values('DATETIME_UTC')
    spe = spe[(spe.DATETIME_UTC >= pd.to_datetime(from_datetime)) & (spe.DATETIME_UTC <= pd.to_datetime(to_datetime))]

    # get all the sensors of the road
    sensors_km = sorted(spe.KM.unique())
    sensors_count = len(sensors_km)

    print('Number of sensors in road {}: {}'.format(road_key, sensors_count))
    #print('Number of events: {}'.format(number_of_events))
    #print('Lanes: {}'.format(number_of_lanes))
    
    plots = []
    plots_info = []
    # add the series
    for km in sensors_km:
        # get the speeds for the sensor
        sensor_speeds = spe[spe.KM == km]
        number_of_lanes = sensor_speeds.LANES.values[0]
        linewidth = number_of_lanes / 2
        
        # events involving the sensor
        sensor_events = sensor_speeds[sensor_speeds['index'].notnull()]
        event_ids = sensor_events['index'].unique() #.astype(int)
        number_of_events = len(event_ids)
        
        # add the speed plot
        plots.append( go.Scatter(x=sensor_speeds.DATETIME_UTC, y=sensor_speeds.SPEED_AVG, name=str(km),
                             mode='lines', marker={ 'line':{ 'width': linewidth } }) )
        # add the events scatter plots if present
        if(number_of_events > 0):
            for evt_id in event_ids:
                e = sensor_events[sensor_events['index'] == evt_id]
                plots.append( go.Scatter(x=e.DATETIME_UTC, y=e.SPEED_AVG, mode='markers', 
                                 name=f'{[evt_id]} {e.EVENT_TYPE.values[0]} (road {e.KM.values[0]})') )
        
        plots_info.append( (str(km), number_of_events+1, number_of_lanes, len(sensor_speeds.DATETIME_UTC) ) )
    
    buttons = []
    total_plots = sum([l[1] for l in plots_info])
    current = 0
    for pi in plots_info:
        buttons.append( { 'label': pi[0],
             'method': 'update',
             'args': [  {'visible': [((j >= current) and (j < current+pi[1])) for j in range(total_plots)] },
                        {'title': f'Sensor {pi[0]} ({pi[2]} lanes) - {pi[3]} data points - {pi[1]-1} events' }
                     ]})
        current += pi[1]
    
    # add the dropdowns
    updatemenus = [{ 'active':-1, 'buttons':buttons }]
    
    iplot({'data':plots, 'layout': {'updatemenus':updatemenus} })
    