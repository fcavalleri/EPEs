import scipy.stats as stats
import xarray as xr
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

path_in = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Series/Pctl/"
path_out = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/Plots/"

rean = 'MERIDA-HRES'

# Define the area of interest
min_lon = 6.0
max_lon = 18.5
min_lat = 35.5
max_lat = 48.0

seasons = ['DJF', 'MAM', 'JJA', 'SON']

# wet hours 50th percentiles

infile_seas = path_in + "MERIDA_HREShourly_wet_hours_50_pctl_seasonal_1986-2022_smooth2_25km.nc"
ds_seas = xr.open_dataset(infile_seas)

seasons = ['DJF', 'MAM', 'JJA', 'SON']

# breaks from 0.9 to 1 with 0.1 step
breaks = np.arange(1,3.75, 0.25)

fig, axs = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()},figsize=(15, 12))
#fig.suptitle('Seasonal 50th percentile (only hours > 1mm, 25km smoothing, 1986-2022)\n', fontsize=35)
axs = axs.ravel()

for season in seasons:
    
    plt.subplot(2, 2, seasons.index(season) + 1)
    axs[seasons.index(season)].set_extent([min_lon, max_lon, min_lat, max_lat])
    axs[seasons.index(season)].add_feature(cfeature.COASTLINE, edgecolor='black')
    # add Italian borders
    axs[seasons.index(season)].add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
    # plot seasonal field with pcolormesh
    cs = ds_seas.pctl_50[seasons.index(season)].plot(ax=axs[seasons.index(season)], transform=ccrs.PlateCarree() , cmap='rainbow', add_colorbar=False, vmin=1.25, vmax=3.75,rasterized=True)
    # add individual colorbar for the subplot
    #cbar = fig.colorbar(cs, ax=axs[seasons.index(season)], orientation='vertical', fraction=0.046, pad=0.04) #ticks=range(0, 225, 25)
    #cbar.set_label('mm/h', fontsize=20, rotation=270, labelpad=20)
    # set season as title
    # do not put any title
    axs[seasons.index(season)].set_title("", fontsize=20)
    axs[seasons.index(season)].text(0.05, 0.05, season, fontsize=20, transform=axs[seasons.index(season)].transAxes,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))

# add common colorbar, thigth layout and save
# set colorbar fraction=0.046, pad=0.04,ticks=range(0, 225, 25) and label "#events" and more lateral space for colorbar ticks
plt.tight_layout()
cbar = fig.colorbar(cs, ax=axs, orientation='vertical', fraction=0.09, pad=0.02)#,ticks=breaks)
cbar.set_label('threshold (mm/h)', fontsize=20, rotation=270, labelpad=20)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=20)
plt.savefig(path_out + 'Figure_1.pdf')
plt.close()











# PCTLS ass to 1 mm

# projetion of ds on the reference g
# Seasonal climatology
infile_seas = path_in + "MERIDA_HREShourly_1mm_pctl_seasonal_1986-2022.nc"
ds_seas = xr.open_dataset(infile_seas)


# breaks from 0.9 to 1 with 0.1 step
breaks = np.arange(0.9, 1.01, 0.1)

fig, axs = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()},figsize=(20, 20))
fig.suptitle('Seasonal percentiles associated to 1 mm (1986-2022)\n', fontsize=35)
axs = axs.ravel()

for season in seasons:
    
    plt.subplot(2, 2, seasons.index(season) + 1)
    axs[seasons.index(season)].set_extent([min_lon, max_lon, min_lat, max_lat])
    axs[seasons.index(season)].add_feature(cfeature.COASTLINE, edgecolor='black')
    # add Italian borders
    axs[seasons.index(season)].add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
    # plot seasonal field with pcolormesh
    cs = ds_seas.pctl_1mm_[seasons.index(season)].plot(ax=axs[seasons.index(season)], transform=ccrs.PlateCarree() , cmap='jet', add_colorbar=False, vmin=0.9, vmax=1.0)
    # add individual colorbar for the subplot
    #cbar = fig.colorbar(cs, ax=axs[seasons.index(season)], orientation='vertical', fraction=0.046, pad=0.04) #ticks=range(0, 225, 25)
    #cbar.set_label('mm/h', fontsize=20, rotation=270, labelpad=20)
    # set season as title
    axs[seasons.index(season)].set_title(season, fontsize=20)

# add common colorbar, thigth layout and save
# set colorbar fraction=0.046, pad=0.04,ticks=range(0, 225, 25) and label "#events" and more lateral space for colorbar ticks
plt.tight_layout()
cbar = fig.colorbar(cs, ax=axs, orientation='vertical', fraction=0.1, pad=0.04)
cbar.set_label('percentile', fontsize=20, rotation=270, labelpad=20)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=20)
plt.savefig(path_out + 'Hourly_PCTLS_1mm.png')
plt.close()



# plot pctls


infile_seas = path_in + "MERIDA_HREShourly_99_pctl_AND1mm_seasonal_1986-2022.nc"
ds_seas = xr.open_dataset(infile_seas)

seasons = ['DJF', 'MAM', 'JJA', 'SON']

# breaks from 0.9 to 1 with 0.1 step
breaks = np.arange(1,2.5, 0.25)

fig, axs = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()},figsize=(20, 20))
fig.suptitle('Seasonal levels associated to 99th percentile ABOVE 1mm (1986-2022)\n', fontsize=35)
axs = axs.ravel()

for season in seasons:
    
    plt.subplot(2, 2, seasons.index(season) + 1)
    axs[seasons.index(season)].set_extent([min_lon, max_lon, min_lat, max_lat])
    axs[seasons.index(season)].add_feature(cfeature.COASTLINE, edgecolor='black')
    # add Italian borders
    axs[seasons.index(season)].add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
    # plot seasonal field with pcolormesh
    cs = ds_seas.pctl_99[seasons.index(season)].plot(ax=axs[seasons.index(season)], transform=ccrs.PlateCarree() , cmap='gist_earth_r', add_colorbar=False, vmin=0, vmax=7)
    # add individual colorbar for the subplot
    #cbar = fig.colorbar(cs, ax=axs[seasons.index(season)], orientation='vertical', fraction=0.046, pad=0.04) #ticks=range(0, 225, 25)
    #cbar.set_label('mm/h', fontsize=20, rotation=270, labelpad=20)
    # set season as title
    axs[seasons.index(season)].set_title(season, fontsize=20)

# add common colorbar, thigth layout and save
# set colorbar fraction=0.046, pad=0.04,ticks=range(0, 225, 25) and label "#events" and more lateral space for colorbar ticks
plt.tight_layout()
cbar = fig.colorbar(cs, ax=axs, orientation='vertical', fraction=0.1, pad=0.04)
cbar.set_label('precipitation level (mm)', fontsize=20, rotation=270, labelpad=20)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=20)
plt.savefig(path_out + 'Hourly_PCTLS_99th_AND1mm.png')
plt.close()


# Seasonal wet hours fraction
infile_seas = path_in + "MERIDA_HREShourly_wet_hours_fraction_seasonal_1986-2022.nc"
ds_seas = xr.open_dataset(infile_seas)


# breaks from 0.9 to 1 with 0.1 step
breaks = np.arange(0, 0.1, 0.0.1)

fig, axs = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()},figsize=(20, 20))
fig.suptitle('Seasonal wet (>1mm) hours fraction (1986-2022)\n', fontsize=35)
axs = axs.ravel()

for season in seasons:
    
    plt.subplot(2, 2, seasons.index(season) + 1)
    axs[seasons.index(season)].set_extent([min_lon, max_lon, min_lat, max_lat])
    axs[seasons.index(season)].add_feature(cfeature.COASTLINE, edgecolor='black')
    # add Italian borders
    axs[seasons.index(season)].add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
    # plot seasonal field with pcolormesh
    cs = ds_seas.wet_fraction[seasons.index(season)].plot(ax=axs[seasons.index(season)], transform=ccrs.PlateCarree() , cmap='jet_r', add_colorbar=False, vmin=0, vmax=0.1)
    # add individual colorbar for the subplot
    #cbar = fig.colorbar(cs, ax=axs[seasons.index(season)], orientation='vertical', fraction=0.046, pad=0.04) #ticks=range(0, 225, 25)
    #cbar.set_label('mm/h', fontsize=20, rotation=270, labelpad=20)
    # set season as title
    axs[seasons.index(season)].set_title(season, fontsize=20)

# add common colorbar, thigth layout and save
# set colorbar fraction=0.046, pad=0.04,ticks=range(0, 225, 25) and label "#events" and more lateral space for colorbar ticks
plt.tight_layout()
cbar = fig.colorbar(cs, ax=axs, orientation='vertical', fraction=0.1, pad=0.04)
cbar.set_label('fraction', fontsize=20, rotation=270, labelpad=20)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=20)
plt.savefig(path_out + 'Hourly_wet_hours_1mm.png')
plt.close()

# range 1 to 2.5 with 0.25 step
breaks = np.arange(1, 2.5, 0.25)



# with spatial averaging
# averaging
window_size = 1
step_size = 0.25

# Define the area of interest
min_lon = 6.0
max_lon = 18.5
min_lat = 35.5
max_lat = 48.0

lon =  np.arange(min_lon, max_lon, step_size)
lat = np.arange(min_lat, max_lat, step_size)
xx, yy = np.meshgrid(lon, lat)

# convert the xarray dataset to a pandas dataframe with lon and lat and tp

for s in 0, 1, 2, 3:  # DJF, MAM, JJA, SON
    df = ds_seas.pctl_50[s].to_dataframe().reset_index()
    # get 1st seasonal dataset

    # lon, lat and tp are the columns of the dataframe
    lon_orig = df['lon'].values
    lat_orig = df['lat'].values
    pctl = df['pctl_50'].values
    # create a new dataframe with the lon, lat and tp columns
    df_cols = pd.DataFrame({'lon': lon_orig, 'lat': lat_orig, 'pctl': pctl})

    # to store the mean values: a pd dataframe
    ds_mean = pd.DataFrame(columns=['lon', 'lat', 'mean'])
    # put lon and lat in the dataframe
    ds_mean['lon'] = xx.flatten()
    ds_mean['lat'] = yy.flatten()

    for i in range(len(xx.flatten())):  #191229
    # print percentual progress every 1000 iterations
        if i % 100 == 0:
            print('Progress: {:.2f}%'.format(i / len(xx.flatten()) * 100))
            # get the lon and lat of the point
        lon_i = ds_mean['lon'][i]
        lat_i = ds_mean['lat'][i]
        # get the tp values in a window of size window_size around the point
        tp_window = df_cols[(df_cols['lon'] >= lon_i - window_size / 2) & (df_cols['lon'] <= lon_i + window_size / 2) &
                            (df_cols['lat'] >= lat_i - window_size / 2) & (df_cols['lat'] <= lat_i + window_size / 2)]
        # calculate the mean of the tp values in the window,remove the nan values
        mean_pctl = tp_window['pctl'].mean()
        # put the mean value in the dataframe
        ds_mean['mean'][i] = mean_pctl

    # save the dataframe to a csv file
    ds_mean.to_csv(path_in + rean + '_wh_pctl_50_1deg_averaged' + seasons[s]  + '.csv', index=False)



#breaks = np.arange(15, 35, 2.5)
breaks = np.arange(1.25, 3.75, 0.1)

fig, axs = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()},figsize=(20, 20))
fig.suptitle('Seasonal levels associated to 50th percentile (without wet hours) (1986-2022)\n', fontsize=35)
axs = axs.ravel()

for season in seasons:
    
    plt.subplot(2, 2, seasons.index(season) + 1)
    axs[seasons.index(season)].set_extent([min_lon, max_lon, min_lat, max_lat])
    axs[seasons.index(season)].add_feature(cfeature.COASTLINE, edgecolor='black')
    # add Italian borders
    axs[seasons.index(season)].add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
    # plot seasonal field with pcolormesh
    ds_mean =  pd.read_csv(path_in + rean + '_wh_pctl_50_1deg_averaged' + season  + '.csv')
    cs = axs[seasons.index(season)].contourf(xx,  yy, 
            ds_mean['mean'].values.reshape(xx.shape), cmap="rainbow",levels=breaks,extend='max')
    # scatterplolot ds_mean['mean'].values, larger dimension point size
    #cs = axs[seasons.index(season)].scatter(ds_mean['lon'], ds_mean['lat'], c=ds_mean['mean'], cmap='jet', s=100, transform=ccrs.PlateCarree(), vmin=1.25, vmax=3.75)
    #cbar = fig.colorbar(cs, ax=axs[seasons.index(season)], orientation='vertical', fraction=0.046, pad=0.04) #ticks=range(0, 225, 25)
    #cbar.set_label('mm/h', fontsize=20, rotation=270, labelpad=20)
    # set season as title
    axs[seasons.index(season)].text(0.05, 0.05, season, fontsize=20, transform=axs[seasons.index(season)].transAxes,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))

# add common colorbar, thigth layout and save
# set colorbar fraction=0.046, pad=0.04,ticks=range(0, 225, 25) and label "#events" and more lateral space for colorbar ticks
plt.tight_layout()
cbar = fig.colorbar(cs, ax=axs, orientation='vertical', fraction=0.1, pad=0.04)
cbar.set_label('precipitation level (mm/h)', fontsize=20, rotation=270, labelpad=20)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=20)
plt.savefig(path_out + 'Hourly_wh_PCTLS_50th_smooth_1deg.png')
plt.close()

