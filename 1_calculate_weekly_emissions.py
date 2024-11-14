

import sys # use print(sys.path) if there is some problem
import warnings
warnings.filterwarnings('ignore')

import util_funcs as uf # this is a file in the directory with some functions

############## LIBRARIES ##############################

import geopandas as gpd
import json
import mobility_airpollution.mobair as mb
import numpy as np
import osmnx as ox  # version '0.16.0' or greater
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

##############################################################################


debug = True
#cities = {"Arezzo", "Carrara", "Grosseto", "Florence", "Livorno", "Lucca", "Massa", "Pisa", "Pistoia", "Prato", "Siena"}
cities = {"Pisa"}

for week in range(22, 23):
     
    for city in cities:
        tdf = pd.read_csv(PATH_TO_INPUT_FILE+'%s_trajectories_week_%s.csv' %(AREA, week))
        
        if debug:
            print("\n\n", tdf.head(1))    
            print("Number of uids in {}: {} \nNumber of rows: {} \n".format(AREA.upper(), tdf['uid'].nunique(), len(tdf)))
        
        region = city + ", Italy"
    
        if debug:
            print("\nCreating tesellation map and trajectory data frame of {} in week {}... \n".format(city, week))
    
        tessellation = uf.download_square_tessellation(cell_size=CELL_SIZE, region=region)
        
        if debug:
            print(tessellation.head(1))
        
        tdf = skmob.TrajDataFrame(tdf)
        
        if debug:
            print("Selecting trajectories within tesellation ...\n")
            
        tdf = uf.select_trajectories_within_tessellation(tdf, tessellation)
        tdf.drop(['tile_ID'], axis=1)  # (dropping column 'tileID', not needed here)
        
        if debug:
            print("Number of uids in {} after applying tesellation map: {} \nNumber of rows left: {} \n".format(city, tdf['uid'].nunique(), len(tdf)))
            print("\nFiltering trajectories on time interval, speed and acceleration\n")
            
        from mobility_airpollution.mobair import filtering
        # if points are distant (t>120), trajectories are split a (5) b (10) c - (120) - d (30) e ( discard d and e if 120 instead of 30)
        # 70 seconds on average
        # we have traj ids, distant parts of the traj counted as separate trajs
        
        # read about mobility and trajectory processing
        # only light vehicles 
        
        tdf_filtered_time = filtering.filter_on_time_interval(tdf, max_interval)

        from mobility_airpollution.mobair import speed
        tdf_with_speed_and_acc = speed.compute_acceleration_from_tdf(tdf_filtered_time)

        ftdf = tdf_with_speed_and_acc.loc[(tdf_with_speed_and_acc['speed'] < 300)]
        ftdf = ftdf.loc[(ftdf['acceleration'] < 10)]
        ftdf = ftdf.loc[(ftdf['acceleration'] > -10)]		              
        # braking not given
        if debug:
            print("Number of uids left in {} after filtering: {} \nNumber of rows after filtering: {} \n".format(city, ftdf['uid'].nunique(), len(ftdf)))
            print("\nCreating directed road network ... (it takes time)")
        
        # loading city's road network from the disk      
        graphml_filename = '%s_network.graphml' % (city.lower())
        #  (This takes a bit longer...)
        road_network_directed = ox.io.load_graphml(filepath=PATH_TO_ROAD_NETWORKS + graphml_filename)
        
        # Taking the undirected version of the network:
        road_network = ox.get_undirected(road_network_directed)

        from mobility_airpollution.mobair import mapmatching
        ftdf_final = mapmatching.find_nearest_edges_in_network(road_network, ftdf, return_tdf_with_new_col=True)

        from mobility_airpollution.mobair import emissions
        import tarfile

        tar = tarfile.open(path_to_table_with_info_on_vehicles, "r:xz")

        for x in tar.getmembers():
            tar_file = tar.extractfile(x)
            modelli_auto = pd.read_csv(tar_file, names=['vid', 'manufacturer', 'type'], usecols = [0,1,2])

        df_emissions = pd.read_csv(path_to_table_with_emission_functions)
        dict_vehicle_fuel_type = emissions.match_vehicle_to_fuel_type(ftdf_final, modelli_auto, ['PETROL', 'DIESEL', 'LPG'])

        tdf_with_emissions = emissions.compute_emissions(ftdf_final, df_emissions, dict_vehicle_fuel_type)
        # add debug here
        
        emissions_totals = tdf_with_emissions[['week', 'week_start', 'uid', 'road_link', 'CO_2', 'NO_x','PM', 'VOC']]
        # add debug if needed
        # this line was added to avoid any problem in grouping step
        #emissions_totals['week'] = emissions_totals['week'].astype(str)   
        emissions_totals = emissions_totals.groupby(['week', 'week_start', 'uid', 'road_link'], as_index=False)[["CO_2", "NO_x","PM", "VOC"]].sum()

        if debug:
            # print the number of lines 
            print(emissions_totals.head(1),"\n\n{}:\nNUMBER OF ROWS {}".format(city.upper(), len(emissions_totals)))
            # print the total number of unique cars 
            print("TOTAL NUMBER OF UNIQUE UIDS: {}".format(emissions_totals['uid'].nunique()))
            # print the total number of unique road links
            print("TOTAL NUMBER OF UNIQUE ROAD LINKS: {}".format(emissions_totals['road_link'].nunique()))

        # saving in csv
        emissions_totals.to_csv(PATH_TO_OUTPUT_FILE+'{}/{}_emissions_week_{}.csv'.format(city.lower(), city.lower(), week), index = False)
        
	



