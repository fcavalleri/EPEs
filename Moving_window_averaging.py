import scipy.stats as stats
import pylab as pl
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import t
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
import xarray as xr

path_out = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/AveragingVals/"
path_in = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/"

rean = "MERIDA-HRES"

years = np.linspace(1986, 2022, 37)  
years = years.astype(int)

# Define the window size and step size (in degrees)
window_size = 0.5  # which is about 15 km, the intrinsic uncertainty of precipitation convection-permitting fields
step_size = 0.1  # which is about 4 km, the resolution of the reanalysis fields

# Define the area of interest
min_lon = 6.0
max_lon = 18.5
min_lat = 35.5
max_lat = 48.0

lon =  np.arange(min_lon, max_lon, step_size)
lat = np.arange(min_lat, max_lat, step_size)
xx, yy = np.meshgrid(lon, lat)

# Define the path to the input files
df_full = pd.DataFrame()

# add each year by rows to the dataframe

for y in years:
    # print processed year
    print(y)
    infile = path_in + "MERIDA_HRES_clusters_pctl_" + str(y) + ".txt"

    df = pd.read_csv(infile, sep=" ")
    df_full = pd.concat([df_full, df], ignore_index=True)

# convert the "time" column to strings
df_full['time'] = df_full['time'].astype(str)
df_full['datetime'] = pd.to_datetime(df_full['time'], format='%Y-%m-%d %H:%M:%S UTC')
df_full['year'] = df_full['datetime'].dt.year
df_full['month'] = df_full['datetime'].dt.month

# also add seasons based on DJF, MAM, JJA, SON
df_full['season'] = (df_full['month']%12 + 3)//3
df_full['season'] = df_full['season'].replace([1, 2, 3, 4], ['DJF', 'MAM', 'JJA', 'SON'])

# add a column "tot_tp_norm" = "tot_tp" / "area"
df_full['tot_tp_norm'] = df_full['tot_tp'] / df_full['area']
# add a column "km_scale" = sqrt((axis_min^2 + axis_max^2)/2) / 0.04
df_full['km_scale'] = df_full['axis_maj'] *110

seasons = ['DJF', 'MAM', 'JJA', 'SON']
vars_to_avg = ["tot_tp_norm", "km_scale", "tp_max"]

df_singlevar = df_full[["year","season","lon_wavg", "lat_wavg", "tot_tp_norm", "km_scale", "tp_max","eccentricity", "time"]]
# change lon_wavg to lon and lat_wavg to lat
df_singlevar.columns = ["year","season","lon", "lat", "tot_tp_norm", "km_scale", "tp_max","eccentricity", "time"]

for seas in seasons:

    df_singlevar_seas = df_singlevar[df_singlevar['season'] == seas]
    
    # create a df with the mean of the values within a moving window
    # cols: lon, lat, 1986, 1987, ..., 2022
    df_counts_avg = pd.DataFrame(columns=["lon", "lat"] + list(years))
    df_km_scale_avg = pd.DataFrame(columns=["lon", "lat"] + list(years))
    df_tp_max_avg = pd.DataFrame(columns=["lon", "lat"] + list(years))
    df_tot_tp_avg = pd.DataFrame(columns=["lon", "lat"] + list(years))
    df_eccentricity_avg = pd.DataFrame(columns=["lon", "lat"] + list(years))
    df_freq_wet = pd.DataFrame(columns=["lon", "lat"] + list(years))

    #means of the values within a moving window window_size = 0.5, step_size = 0.1
    for lon in np.arange(min_lon, max_lon, step_size):
        for lat in np.arange(min_lat, max_lat, step_size):

            print(seas + " lon: " + str(round(lon, 2)) + "/" + str(max_lon) + " lat: " + str(round(lat, 2)) + "/" + str(max_lat))
            # create a mask for the values within the moving window
            mask = (df_singlevar_seas['lon'] >= lon) & (df_singlevar_seas['lon'] < lon + window_size) & (df_singlevar_seas['lat'] >= lat) & (df_singlevar_seas['lat'] < lat + window_size)
            # create a vector to store annual mean values
            freq_wet = np.zeros(len(years))
            counts = np.zeros(len(years))
            km_scale_avg = np.zeros(len(years))
            tp_max_avg = np.zeros(len(years))
            tot_tp_avg = np.zeros(len(years))
            eccentricity_avg = np.zeros(len(years))

            for i, year in enumerate(years):
                # create a mask for the values within the moving window and the year
                mask_year = (df_singlevar_seas['year'] == year)
                
                # number of unique times in the moving window and the year
                freq_wet[i] = df_singlevar_seas[mask & mask_year]['time'].nunique() /(24*n_days_in_season[seasons.index(seas)])  # number of unique times in the moving window and the year divided by the number of hours in the season
                # calculate the number of elements within the moving window and the year
                counts[i] = df_singlevar_seas[mask & mask_year]['tot_tp_norm'].count()
                km_scale_avg[i] = df_singlevar_seas[mask & mask_year]['km_scale'].mean()
                tp_max_avg[i] = df_singlevar_seas[mask & mask_year]['tp_max'].mean()
                tot_tp_avg[i] = df_singlevar_seas[mask & mask_year]['tot_tp_norm'].mean()
                eccentricity_avg[i] = df_singlevar_seas[mask & mask_year]['eccentricity'].mean()

            # concat the mean vector to the dataframe in the column year
            df_counts_avg = pd.concat([df_counts_avg, pd.DataFrame({"lon": [lon + window_size / 2], "lat": [lat + window_size / 2], **{year: [counts[i]] for i, year in enumerate(years)}})], ignore_index=True)
            df_km_scale_avg = pd.concat([df_km_scale_avg, pd.DataFrame({"lon": [lon + window_size / 2], "lat": [lat + window_size / 2], **{year: [km_scale_avg[i]] for i, year in enumerate(years)}})], ignore_index=True)
            df_tp_max_avg = pd.concat([df_tp_max_avg, pd.DataFrame({"lon": [lon + window_size / 2], "lat": [lat + window_size / 2], **{year: [tp_max_avg[i]] for i, year in enumerate(years)}})], ignore_index=True)
            df_tot_tp_avg = pd.concat([df_tot_tp_avg, pd.DataFrame({"lon": [lon + window_size / 2], "lat": [lat + window_size / 2], **{year: [tot_tp_avg[i]] for i, year in enumerate(years)}})], ignore_index=True)
            df_eccentricity_avg = pd.concat([df_eccentricity_avg, pd.DataFrame({"lon": [lon + window_size / 2], "lat": [lat + window_size / 2], **{year: [eccentricity_avg[i]] for i, year in enumerate(years)}})], ignore_index=True)
            df_freq_wet = pd.concat([df_freq_wet, pd.DataFrame({"lon": [lon + window_size / 2], "lat": [lat + window_size / 2], **{year: [freq_wet[i]] for i, year in enumerate(years)}})], ignore_index=True)

    # save the df_averaged as txt
    df_counts_avg.to_csv(path_out + 'pctl_eachyr_Seas_EventBased_Mean_counts_' + seas + '.txt', sep=' ', index=False)
    df_km_scale_avg.to_csv(path_out + 'pctl_eachyr_Seas_EventBased_Mean_km_scale_' + seas + '.txt', sep=' ', index=False)
    df_tp_max_avg.to_csv(path_out + 'pctl_eachyr_Seas_EventBased_Mean_tp_max_' + seas + '.txt', sep=' ', index=False)
    df_tot_tp_avg.to_csv(path_out + 'pctl_eachyr_Seas_EventBased_Mean_tot_tp_' + seas + '.txt', sep=' ', index=False)
    df_eccentricity_avg.to_csv(path_out + 'pctl_eachyr_Seas_EventBased_Mean_eccentricity_' + seas + '.txt', sep=' ', index=False)
    df_freq_wet.to_csv(path_out + 'pctl_eachyr_Seas_EventBased_Mean_freq_wet_' + seas + '.txt', sep=' ', index=False)
