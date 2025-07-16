import scipy.stats as stats
import pylab as pl
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pandas import DataFrame
from pandas import to_datetime
from pandas import read_csv
from pandas import Series
from pandas import concat
from EventsStatistics_functions import MovingWindowSlope
import matplotlib.ticker as mticker

rean = 'MERIDA-HRES'
path_in = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/AveragingVals/"
path_out = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/Plots/"

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

# get lons lats from pd.read_csv(path_in + 'EXT_eachyr_Seas_EventBased_Mean_counts_JJA.txt', sep=' ')
mat_template = pd.read_csv(path_in + '24_EXTR_Eachyr_Seas_EventBased_Mean_counts_JJA.txt', sep=' ') 
lats = mat_template.iloc[:, 1].values
lons = mat_template.iloc[:, 0].values
lats = np.unique(lats)
lons = np.unique(lons)
xx, yy = np.meshgrid(lons, lats)

vars_to_clim = ["counts","tot_tp", "km_scale", "tp_max","eccentricity","freq_wet"]
udms = ["#events","mm/day", "km", "mm/day", "eccentricity",""]
names = ["Number of events", "Average intensity", "Peak intensity", "Spatial scale","Eccentricity","Frequency of wet hours"]

seasons = ['DJF', 'MAM', 'JJA', 'SON']

#### TOT_TP_NORM
var = 'tot_tp'
breaks = np.linspace(1,10, 10)
#breaks = np.arange(7,23,2)

print(var)
# same figure with 4 seasons
fig, axs = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()},figsize=(14, 12))
axs = axs.ravel()

for season in seasons:
    #load matrix saved as txt at df_counts_avg.to_csv(path_out + 'EXT_eachyr_Seas_EventBased_Mean_counts_' + seas + '.txt', sep=' ', index=False)
    mat_seas = pd. read_csv(path_in + 'pctl_Eachyr_Seas_EventBased_Mean_' + var + '_' + season + '.txt', sep=' ')
    # avg each row exluded lon and lat
    clim_val = np.nanmean(mat_seas.iloc[:, 2:], axis=1)

    plt.subplot(2, 2, seasons.index(season) + 1)
    axs[seasons.index(season)].set_extent([min_lon+0.25, max_lon, min_lat+0.25, max_lat])
    axs[seasons.index(season)].add_feature(cfeature.COASTLINE, edgecolor='black')
    # add Italian borders
    axs[seasons.index(season)].add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
    cs = axs[seasons.index(season)].contourf(xx,  yy, 
            clim_val.reshape(xx.shape).T, cmap="gist_earth_r",levels=breaks,extend='max')
     axs[seasons.index(season)].text(0.05, 0.05, season, fontsize=20, transform=axs[seasons.index(season)].transAxes,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))

plt.tight_layout()
cbar = fig.colorbar(cs, ax=axs, orientation='vertical', fraction=0.09, pad=0.02)#,ticks=breaks)
cbar.set_label("AvIn (mm/h)", fontsize=20, rotation=270, labelpad=20)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=20)
plt.savefig(path_out + 'pctl_climatology_' + var + '_seas.pdf')
plt.close()

