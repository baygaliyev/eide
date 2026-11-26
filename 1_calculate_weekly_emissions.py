"""Step 1: weekly road-level emissions from vehicular GPS traces.

In plain terms, this script takes restricted weekly trajectory files and an
OpenStreetMap road network, lays a grid over the area of interest, keeps the
trips inside that grid, matches them to road links, and uses the `mobair`
package to estimate speed, acceleration and CO2/NOx/PM/VOC emissions per link.
Results are written week by week to `data/emissions/`. The trajectories and
road-network files are not distributed with this repository; see the README.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import util_funcs as uf

############## LIBRARIES ##############################

import geopandas as gpd
import json
import mobair as mb
import numpy as np
import osmnx as ox  # see requirements.txt for the supported 1.x range
import pandas as pd
import psycopg2
import skmob as skmob
from skmob.utils.plot import plot_gdf
from skmob.tessellation import tilers
#######################################################


####################### PARAMETERS ###########################################
AREA = 'italy'
CELL_SIZE = 1500 # size of cell in the spatial tessellation

max_interval = 120 # seconds

PATH_TO_ROAD_NETWORKS = './data/road_networks/'
PATH_TO_INPUT_FILE = './data/trajectories/'
PATH_TO_OUTPUT_FILE = './data/emissions/'

path_to_table_with_info_on_vehicles = './data/modelli_auto.tar.xz'
path_to_table_with_emission_functions = './data/emission_functions.csv'

debug = True
cities = {"Pisa"}

for week in range(22, 23):
     
    for city in cities:
        tdf = pd.read_csv(PATH_TO_INPUT_FILE+'%s_trajectories_week_%s.csv' %(AREA, week))
        
        if debug:
            print("\n\n", tdf.head(1))    
            print("Number of uids in {}: {} \nNumber of rows: {} \n".format(AREA.upper(), tdf['uid'].nunique(), len(tdf)))
        
        region = city + ", Italy"
    
        if debug:
            print("\nCreating tessellation map and trajectory data frame of {} in week {}... \n".format(city, week))
    
        tessellation = uf.download_square_tessellation(cell_size=CELL_SIZE, region=region)
        
        if debug:
            print(tessellation.head(1))
        
        tdf = skmob.TrajDataFrame(tdf)
        
        if debug:
            print("Selecting trajectories within tessellation ...\n")
            
        tdf = uf.select_trajectories_within_tessellation(tdf, tessellation)
        if debug:
            print("Number of uids in {} after applying tessellation map: {} \nNumber of rows left: {} \n".format(city, tdf['uid'].nunique(), len(tdf)))
            print("\nFiltering trajectories on time interval, speed and acceleration\n")
            
        from mobair import filtering
        # Split trajectories at time gaps longer than `max_interval` and keep
        # only light vehicles; distant parts count as separate trajectories.
        
        tdf_filtered_time = filtering.filter_on_time_interval(tdf, max_interval)

        from mobair import speed
        tdf_with_speed_and_acc = speed.compute_acceleration_from_tdf(tdf_filtered_time)

        ftdf = tdf_with_speed_and_acc.loc[(tdf_with_speed_and_acc['speed'] < 300)]
        ftdf = ftdf.loc[(ftdf['acceleration'] < 10)]
        ftdf = ftdf.loc[(ftdf['acceleration'] > -10)]		              
        # Bound implausible acceleration values.
        if debug:
            print("Number of uids left in {} after filtering: {} \nNumber of rows after filtering: {} \n".format(city, ftdf['uid'].nunique(), len(ftdf)))
            print("\nCreating directed road network ... (it takes time)")
        
        # Load the city's road network from disk.
        graphml_filename = '%s_network.graphml' % (city.lower())
        road_network_directed = ox.io.load_graphml(filepath=PATH_TO_ROAD_NETWORKS + graphml_filename)
        
        # Use the undirected network for map matching.
        road_network = ox.get_undirected(road_network_directed)

        from mobair import mapmatching
        ftdf_final = mapmatching.find_nearest_edges_in_network(road_network, ftdf, return_tdf_with_new_col=True)

        from mobair import emissions
        import tarfile

        tar = tarfile.open(path_to_table_with_info_on_vehicles, "r:xz")

        for x in tar.getmembers():
            tar_file = tar.extractfile(x)
            modelli_auto = pd.read_csv(tar_file, names=['vid', 'manufacturer', 'type'], usecols = [0,1,2])

        df_emissions = pd.read_csv(path_to_table_with_emission_functions)
        dict_vehicle_fuel_type = emissions.match_vehicle_to_fuel_type(ftdf_final, modelli_auto, ['PETROL', 'DIESEL', 'LPG'])

        tdf_with_emissions = emissions.compute_emissions(ftdf_final, df_emissions, dict_vehicle_fuel_type)
        emissions_totals = tdf_with_emissions[['week', 'week_start', 'uid', 'road_link', 'CO_2', 'NO_x','PM', 'VOC']]
        emissions_totals = emissions_totals.groupby(['week', 'week_start', 'uid', 'road_link'], as_index=False)[["CO_2", "NO_x","PM", "VOC"]].sum()

        if debug:
            print(emissions_totals.head(1),"\n\n{}:\nNUMBER OF ROWS {}".format(city.upper(), len(emissions_totals)))
            print("TOTAL NUMBER OF UNIQUE UIDS: {}".format(emissions_totals['uid'].nunique()))
            print("TOTAL NUMBER OF UNIQUE ROAD LINKS: {}".format(emissions_totals['road_link'].nunique()))

        # Write weekly totals to disk.
        emissions_totals.to_csv(PATH_TO_OUTPUT_FILE+'{}/{}_emissions_week_{}.csv'.format(city.lower(), city.lower(), week), index = False)
        
	



