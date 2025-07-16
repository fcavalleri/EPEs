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

# load clusters_mean from the csv file
clusters_mean = pd.read_csv(path_in + rean + '_pctl_EXTR_clusters_mean_1986-2022_JJA.txt', sep=' ', index_col=0)

# clusters_mean rownames to float
clusters_mean.index = clusters_mean.index.astype(float)
# save them to xx
yy = clusters_mean.index.values
# clusters_mean column names to float
clusters_mean.columns = clusters_mean.columns.astype(float)
# save them to yy
xx = clusters_mean.columns.values

breaks= np.arange(1, 1.6, 0.1)

seasons = ['JJA', 'SON']

fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()},figsize=(14, 12))
#fig.suptitle('Seasonal climatology of '+ var, fontsize=35)
axs = axs.ravel()   

for i, season in enumerate(seasons):
    # load clusters_mean from the csv file
    clusters_mean = pd.read_csv(path_in + rean + '_pctl_EXTR_clusters_mean_1986-2022_' + season + '.txt', sep=' ', index_col=0)
    # convert clusters_mean to a numpy array
    # save clusters_mean to netcdf file
    plt.subplot(1, 2, seasons.index(season) + 1)
    axs[seasons.index(season)].set_extent([min_lon+0.25, max_lon-0.25, min_lat+0.25, max_lat-0.25])
    axs[seasons.index(season)].add_feature(cfeature.COASTLINE, edgecolor='black')
    # add Italian borders
    axs[seasons.index(season)].add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
    # plot seasonal field with pcolormesh
    cs = axs[seasons.index(season)].contourf(xx, yy, clusters_mean.T, cmap='viridis', levels=breaks, extend='max', shading='auto')
    axs[seasons.index(season)].text(0.05, 0.05, season, fontsize=20, transform=axs[seasons.index(season)].transAxes,
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))

plt.tight_layout()
plt.subplots_adjust(right=0.9)
cbar = fig.colorbar(cs, ax=axs, orientation='vertical', fraction=0.05, pad=0.02,ticks=breaks)
cbar.set_label('average EPEs persistence (hours)', fontsize=20, rotation=270, labelpad=20)
cbar.ax.set_aspect(8)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=15)
plt.savefig(path_out + 'pctls_avgEPEs_persistence_1986-2022_JJASON.pdf')
