import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pandas import DataFrame
from pandas import to_datetime
from pandas import read_csv
from pandas import Series
from pandas import concat
import matplotlib as mpl
import xarray as xr

rean = "MERIDA-HRES"
path_in = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/"

path_extremes = "/media/clima_out_c1/DATI/DataForValidation/Reanalyses/MERIDA-HRES/tp/Extremes/"
ymax_1h = path_extremes + "Annual_extremes_MERIDA_HRES_1h_max_clim_1986-2023_2smooth20km.nc"
#ymax_24h = path_extremes + "Annual_extremes_MERIDA_HRES_24h_max_clim_1986-2023.nc"

path_out = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/Plots/"

#path_extremes = "/media/clima_out_c1/DATI/DataForValidation/Reanalyses/MOLOCH/tp/Extremes/"
#ymax_1h = path_extremes + "Annual_extremes_moloch_1h_max_clim_1986-2022.nc"
#path_out = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/Plots/"

years = np.linspace(1986, 2022, 37)  # 1986-2020
years = years.astype(int)

# Define the window size and step size (in degrees)
window_size = 0.5  # which is about 15 km, the intrinsic uncertainty of precipitation convection-permitting fields
step_size = 0.125  # which is about 4 km, the resolution of the reanalysis fields

# Define the area of interest
min_lon = 6.125
max_lon = 18.375
min_lat = 35.625
max_lat = 47.825

lon =  np.arange(min_lon, max_lon, step_size)
lat = np.arange(min_lat, max_lat, step_size)
xx, yy = np.meshgrid(lon, lat)

# Define the path to the input files
df_extreme = pd.read_csv(path_in + rean + '_pctl_extremes_1986-2022.txt', sep=' ')

seasons = ['JJA', 'SON']

for season in seasons:
    # filter df_extreme by season
    df_season = df_extreme[df_extreme['season'] == season]
    df_season['diff'] = pd.to_datetime(df_season['time'], format='%Y-%m-%d %H:%M:%S UTC').diff().dt.total_seconds() / 3600

    window_size = 0.5  
    step_size = 0.125

    # Define the area of interest
    min_lon = 6.0
    max_lon = 19.5
    min_lat = 35.5
    max_lat = 48.0

    lon =  np.arange(min_lon+step_size, max_lon-step_size, step_size)
    lat = np.arange(min_lat+step_size, max_lat-step_size, step_size)
    xx, yy = np.meshgrid(lon, lat)

    #cluster func

    def time_clustering(v):
        clusters = []
        i = 0
        n = len(v)

        while i < n:
            if v[i] == 1:
                start = i
                # Estendi il cluster finché trovi 1
                while i + 1 < n and v[i + 1] == 1:
                    i += 1
                end = i

                # Aggiungi l'elemento precedente se esiste
                cluster_start = start - 1 if start > 0 else start
                cluster = list(range(cluster_start, end + 1))
                clusters.append(cluster)
            i += 1

        return clusters

    # matrix to save mean cluster length
    clusters_mean = np.zeros((len(lon), len(lat)))

    for i in range(len(lon)):
        for j in range(len(lat)):
            print("lon: ", lon[i], "lat: ", lat[j])
            # create a mask for the points within the window
            mask = (df_season['lon_max'] >= lon[i]) & (df_season['lon_max'] < lon[i] + window_size) & \
                (df_season['lat_max'] >= lat[j]) & (df_season['lat_max'] < lat[j] + window_size)
            
            # how much events are in the window
            n_events = len(df_season[mask])
            diff_local = pd.to_datetime(df_season[mask]['time'], format='%Y-%m-%d %H:%M:%S UTC').diff().dt.total_seconds() / 3600
            # convert to integers
            diff_local = diff_local.fillna(0).astype(int)
            # get the indices of the events in the window
            full_indices = df_season[mask].index
            # clustering
            clusters = time_clustering(diff_local.values)
            # how much clusters are in the window
            n_clusters = len(clusters)
            # how much element in each cluster
            n_elements = [len(cluster) for cluster in clusters]
            total_elements = sum(n_elements)
            # concatenate n_elements with 1 repeated for n_events - total_elements times
            duration = np.concatenate((n_elements, np.ones(n_events - total_elements)))
            clusters_mean[i, j] = np.mean(duration)
            # print timestamp of the events in the cluster
            #for cluster in clusters:
            #    print("Cluster: ", cluster)
            #    for k in cluster:
            #        print(df_extreme[mask]['time'].values[k])
            #   print("")

            # for each cluster, retain only the event with the maximum tp
            #for n in range(len(clusters)):
            #    cluster = clusters[n]
            #    # get the events withcthe index given by cluster
            #    tp_max_vals = df_extreme[mask]['tp_max']
            #    tp_max_vals = tp_max_vals.index == cluster
                
                # get the index of the event with the maximum tp
                #keep_index = cluster[max_tp_index]
                # remove all other events in the cluster
                #for k in cluster:
                #    if k != max_tp_index:
                #        df_extreme = df_extreme.drop(df_extreme[mask].index[k])

    # save clusters_mean to a csv file
    clusters_mean_df = pd.DataFrame(clusters_mean, columns=lon, index=lat)
    clusters_mean_df.to_csv(path_in + rean + '_pctl_EXTR_clusters_mean_1986-2022_' + season + '.txt', sep=' ', index=True)

