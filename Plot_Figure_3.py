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

rean = 'MERIDA-HRES'

path_extremes = "/media/clima_out_c1/DATI/DataForValidation/Reanalyses/MERIDA-HRES/tp/Extremes/"
ymax_1h = path_extremes + "Annual_extremes_MERIDA_HRES_1h_max_clim_1986-2023_2smooth20km.nc"

path_out = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/Plots/"

years = np.linspace(1986, 2022, 37)
years = years.astype(int)

# Define the area of interest
min_lon = 6.0
max_lon = 18.5
min_lat = 35.5
max_lat = 48.0

lon =  np.arange(min_lon, max_lon, step_size)
lat = np.arange(min_lat, max_lat, step_size)
xx, yy = np.meshgrid(lon, lat)

# Open netcdf file with xarray

ds = xr.open_dataset(ymax_1h)

# plot ds with also ax.add_feature(cfeature.COASTLINE, edgecolor='white')
# add Italian borders

plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
# plot the data with pcolormesh but NO COLORBAR
cs = ds['tp'].plot(ax=ax, transform=ccrs.PlateCarree(), cmap='rainbow',add_colorbar=False,rasterized=True)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.set_extent([6, 18.5, 35.5, 48])
plt.title('')
# add the colorbar
cbar = plt.colorbar(cs,ax=ax, orientation='vertical', fraction=0.09, pad=0.02)#,ticks=breaks)
cbar.set_label('average RX1hour (mm/h)', fontsize=15, rotation=270, labelpad=20)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=15)
plt.savefig(path_out + 'Figure_3.pdf')
plt.close()

