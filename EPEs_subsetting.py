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

path_out = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/Plots/"

years = np.linspace(1986, 2022, 37)
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

# Load the data in the netCDF file
ds = xr.open_dataset(ymax_1h)

# create a dataset with lon, lat and tp as columns from ds
df_thr = ds['tp'].to_dataframe().reset_index()
#drop nan values
df_thr = df_thr.dropna()

# find df min values 
thr_min = df_thr['tp'].min()

# drop df_full with tp_max values below thr_min
df_start = df_full
df_full = df_full[df_full['tp_max'] >= thr_min]

# NN procedure

# extract lon_wavg column from df_full
lon_events = df_full['lon_max'].values
# extract unique lon values from df_thr
lon_thr = ds['lon'].values
# find the nearest values of lon_thr to lon_events
#lon_grid = np.array([lon_thr[np.abs(lon_thr - lon).argmin()] for lon in lon_events])
lon_grid_idx =  np.array([[np.abs(lon_thr - lon).argmin()] for lon in lon_events])

lat_events = df_full['lat_max'].values
lat_thr = ds['lat'].values
#lat_grid = np.array([lat_thr[np.abs(lat_thr - lat).argmin()] for lat in lat_events])
lat_grid_idx =  np.array([[np.abs(lat_thr - lat).argmin()] for lat in lat_events])

# ds to a numpy 2d matrix
tp_thr_mat = ds['tp'].values
#its dimensions
print(tp_thr_mat.shape)
# convert into a 2d array
tp_thr_mat = tp_thr_mat.reshape((tp_thr_mat.shape[0]*tp_thr_mat.shape[1], tp_thr_mat.shape[2]))
print(tp_thr_mat.shape)

# get tp value from df_thr based on lon_grid and lat_grid
tp_max_year = np.array([tp_thr_mat[lat_grid_idx[i], lon_grid_idx[i]] for i in range(len(df_full))])

# add tp_max to df_full
df_full['tp_max_thr'] = tp_max_year

# create a subset df with only the events with tp_max > tp_max_thr
df_extreme = df_full[df_full['tp_max'] > df_full['tp_max_thr']]

# save df_extreme to a txt file
df_extreme.to_csv(path_in + rean + '_pctl_extremes_1986-2022.txt', sep=' ', index=False)



# General statistics

n_events_tot = len(df_start['lon_max'].values)
n_events_extreme = len(df_extreme['lon_max'].values)
print(n_events_extreme / n_events_tot)

# event ration season by season
df_season = df_start.groupby('season').size().reset_index(name='counts')
df_season_extreme = df_extreme.groupby('season').size().reset_index(name='counts')
n_events_season = df_season['counts'].values
n_events_season_extreme = df_season_extreme['counts'].values
print(n_events_season_extreme / n_events_season)
