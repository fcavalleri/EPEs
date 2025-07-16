import pandas as pd
import scipy as sp
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from pandas import DataFrame
from pandas import to_datetime
from pandas import read_csv
from pandas import Series
from pandas import concat
import matplotlib as mpl
import xarray as xr
import pymannkendall as mk
from scipy.stats import rankdata
import statsmodels.stats.multitest as statsmodels
import matplotlib.ticker as mticker
import proplot as pplt

rean = "MERIDA-HRES"

formatter = mticker.FuncFormatter(lambda x, pos: f'{x*100:.0f}%')

#path_in = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/AveragingVals/"
path_in = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/AveragingVals/"
path_out = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/Plots/"

def rel_change(values,slope):
        clim_val = np.nanmean(values)
        normed_values = slope / clim_val
        decadal_slope = normed_values * 10  # convert to decadal slope

        return decadal_slope

years = np.linspace(1986, 2022, 37)
#years = np.linspace(2000,2022, 23)
years = years.astype(int)

# Define the window size and step size (in degrees)
window_size = 0.5  # which is about 15 km, the intrinsic uncertainty of precipitation convection-permitting fields
step_size = 0.1  # which is about 4 km, the resolution of the reanalysis fields

# Define the area of interest
min_lon = 6.0
max_lon = 19.0
min_lat = 35.5
max_lat = 48.0

# get lons lats from pd.read_csv(path_in + 'EXT_eachyr_Seas_EventBased_Mean_counts_JJA.txt', sep=' ')
mat_template = pd.read_csv(path_in + 'EXT_eachyr_Seas_EventBased_Mean_counts_JJA.txt', sep=' ') 
lats = mat_template.iloc[:, 1].values
lons = mat_template.iloc[:, 0].values
# get unique lats and lons
lats = np.unique(lats)
lons = np.unique(lons)
xx, yy = np.meshgrid(lons, lats)


vars_to_trend = ["counts","tot_tp", "km_scale", "tp_max"]
udms = ["#events","mm/h", "km", "mm/h"]
names = ["Number of events", "Average intensity", "Peak intensity", "Spatial scale"]

## By season
seasons = ['JJA', 'SON']


## TRENDS PLOTTING

#for var in vars_to_trend:
var="counts"

# box lon lat centers
box_lons_JJA = [7.6,11,16.3,14.9]
box_lats_JJA = [45.5, 46.4,38.9,37]

box_lons_SON = [16.5,9.5,9.5,12.1]
box_lats_SON = [40.5,40.7,44.2,45.7]

# get 4 colors from the cmap ACCENT
series_colors =  [
        (1.0, 0.4, 0.0),      # Arancione chiaro
        (1.0, 0.7, 0.0),      # Arancione scuro
        (1.0, 0.0, 0.0),      # Rosso
         (0.3, 0.3, 0.3),      # Grigio chiaro
        (1.0, 0.0, 1.0),      # Viola
        "blue",      # Azzurro acceso
        (0.5, 0.0, 0.5),      # Fucsia chiaro
        (0.3, 0.3, 0.3)     # Grigio scuro
]

print(var)
breaks = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5]


##### Figure 13

fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()},figsize=(14, 6))
#fig, axs = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()},figsize=(14, 10))
#fig.suptitle('Decadal relative trend of '+ var, fontsize=35)
axs = axs.ravel()

for season in seasons:
    # load the slopes_norm_masked
    slopes_norm_masked = np.loadtxt(path_in + 'pctl_EXTR_' + var + '_slopes_norm_' + season + '.txt')
    plt.subplot(1, 2, seasons.index(season) + 1)
    axs[seasons.index(season)].set_extent([min_lon+0.25, max_lon-0.35, min_lat+0.25, max_lat-0.25])
    axs[seasons.index(season)].add_feature(cfeature.COASTLINE, edgecolor='black')
    # add Italian borders
    axs[seasons.index(season)].add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
    # plot seasonal field with pcolormesh
    #cs = axs[seasons.index(season)].contourf(lons.reshape(xx.shape), lats.reshape(xx.shape), 
                #slopes.reshape(xx.shape).T, cmap='rainbow', levels=100,extend='both')  #levels=np.linspace(50, 400, 8)
    cs = axs[seasons.index(season)].contourf(xx,  yy, 
            slopes_norm_masked.reshape(xx.shape).T, cmap="BrBG",levels=breaks,extend='both')
                # tot tp norm: cmap="winter_r", levels=np.linspace(1.5,4, 6),extend='both')
    signif_mask = np.loadtxt(path_in + 'pctl_EXTR_' + var + '_signif_mask_' + season + '.txt')
    # plot black dots where signif_mask is 1
    axs[seasons.index(season)].scatter(xx[signif_mask.reshape(xx.shape).T==1], yy[signif_mask.reshape(xx.shape).T==1], color='black', s=0.5, marker="x" ,transform=ccrs.PlateCarree())
    # load zero_years and mask with grey contourf where zero_years is > 10
    zero_years = np.loadtxt(path_in + 'pctl_EXTR_counts_zero_years_' + season + '.txt')
    # put zero_years to 1 where is > 10, and 0 where is <= 10
    zero_years = np.where(zero_years > 10, 1, 0) 
    # grey contourf ONLY where zero_years is 1, trasparent where zero_years is 0
    axs[seasons.index(season)].contourf(xx, yy,
            zero_years.reshape(xx.shape).T, cmap="Greys", levels=[0.1, 0.5, 1], extend='none', alpha=0.6)
    # add a 0.5 degree box centered on box_lons[1] and box_lats[1]
    if season == 'JJA':
        axs[seasons.index(season)].add_patch(plt.Rectangle((box_lons_JJA[0]-0.25, box_lats_JJA[0]-0.25), 0.5
                , 0.5, linewidth=3, edgecolor=series_colors[0], facecolor='none', transform=ccrs.PlateCarree()))
        axs[seasons.index(season)].text(box_lons_JJA[0]-0.6, box_lats_JJA[0]-0.2, '1', fontsize=20, color=series_colors[0], transform=ccrs.PlateCarree())
        axs[seasons.index(season)].add_patch(plt.Rectangle((box_lons_JJA[1]-0.25, box_lats_JJA[1]-0.25), 0.5
                , 0.5, linewidth=3, edgecolor=series_colors[1], facecolor='none', transform=ccrs.PlateCarree()))
        axs[seasons.index(season)].text(box_lons_JJA[1]-0.6, box_lats_JJA[1]-0.2, '2', fontsize=20, color=series_colors[1], transform=ccrs.PlateCarree())
        axs[seasons.index(season)].add_patch(plt.Rectangle((box_lons_JJA[2]-0.25, box_lats_JJA[2]-0.25), 0.5
                , 0.5, linewidth=3, edgecolor=series_colors[2], facecolor='none', transform=ccrs.PlateCarree()))
        axs[seasons.index(season)].text(box_lons_JJA[2]-0.6, box_lats_JJA[2]-0.2, '3', fontsize=20, color=series_colors[2], transform=ccrs.PlateCarree())
        axs[seasons.index(season)].add_patch(plt.Rectangle((box_lons_JJA[3]-0.25, box_lats_JJA[3]-0.25), 0.5
                , 0.5, linewidth=3, edgecolor=series_colors[3], facecolor='none', transform=ccrs.PlateCarree()))
        axs[seasons.index(season)].text(box_lons_JJA[3]-0.6, box_lats_JJA[3]-0.2, '4', fontsize=20, color=series_colors[3], transform=ccrs.PlateCarree())
    if season == 'SON':
        axs[seasons.index(season)].add_patch(plt.Rectangle((box_lons_SON[0]-0.25, box_lats_SON[0]-0.25), 0.5
                , 0.5, linewidth=3, edgecolor=series_colors[4], facecolor='none', transform=ccrs.PlateCarree()))
        axs[seasons.index(season)].text(box_lons_SON[0]-0.6, box_lats_SON[0]-0.2, '5', fontsize=20, color=series_colors[4], transform=ccrs.PlateCarree())
        axs[seasons.index(season)].add_patch(plt.Rectangle((box_lons_SON[1]-0.25, box_lats_SON[1]-0.25), 0.5
                , 0.5, linewidth=3, edgecolor=series_colors[5], facecolor='none', transform=ccrs.PlateCarree()))
        axs[seasons.index(season)].text(box_lons_SON[1]-0.6, box_lats_SON[1]-0.2, '6', fontsize=20, color=series_colors[5], transform=ccrs.PlateCarree())
        axs[seasons.index(season)].add_patch(plt.Rectangle((box_lons_SON[2]-0.25, box_lats_SON[2]-0.25), 0.5
                , 0.5, linewidth=3, edgecolor=series_colors[6], facecolor='none', transform=ccrs.PlateCarree()))
        axs[seasons.index(season)].text(box_lons_SON[2]-0.6, box_lats_SON[2]-0.2, '7', fontsize=20, color=series_colors[6], transform=ccrs.PlateCarree())
        axs[seasons.index(season)].add_patch(plt.Rectangle((box_lons_SON[3]-0.25, box_lats_SON[3]-0.25), 0.5
                , 0.5, linewidth=3,edgecolor=series_colors[7], facecolor='none', transform=ccrs.PlateCarree()))
        axs[seasons.index(season)].text(box_lons_SON[3]-0.6, box_lats_SON[3]-0.2, '8', fontsize=20, color=series_colors[7], transform=ccrs.PlateCarree())

    axs[seasons.index(season)].text(0.05, 0.05, season, fontsize=20, transform=axs[seasons.index(season)].transAxes,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))

# add common colorbar, thigth layout and save
# set colorbar fraction=0.046, pad=0.04,ticks=range(0, 225, 25) and label "#events" and more lateral space for colorbar ticks
plt.tight_layout()  # se la colorbar è a destra
cbar = fig.colorbar(cs, ax=axs, orientation='vertical', fraction=0.09, pad=0.02,ticks=breaks)
cbar.set_label("decadal change", fontsize=20, rotation=270, labelpad=20)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_yticklabels(['{:,.0%}'.format(x) for x in breaks])
#cbar.ax.yaxis.set_major_formatter(formatter)
plt.savefig(path_out + 'pctl_EXTR_trend_mask_boxes_' + var + '_seas.pdf')
plt.close()


##### Figure 14


fig, axs = plt.subplots(4, 2, figsize=(10, 8))

season = 'JJA'
mat_seas = pd.read_csv(path_in + 'pctl_EXTR_eachyr_Seas_EventBased_Mean_counts_JJA.txt', sep=' ')
for i in range(len(box_lons_JJA)):
    plt.subplot(4, 2, 2*i + 1)
    if i == 0:
        axs[i, 0].set_title(f'JJA', fontsize=20)

    axs[i, 0].set_xlim(1986, 2022)
    axs[i, 0].set_ylim(0, 52)
    # set y ticks each 10
    axs[i, 0].yaxis.set_major_locator(mticker.MultipleLocator(10))
    axs[i, 0].grid(True, linestyle='--', alpha=0.5)
    #axs[i, 0].set_xlabel('Year', fontsize=16)
    axs[i, 0].set_ylabel('N EPEs / season', fontsize=12)
    # create a mask for mat_seas['lat'] between box_lats_JJA[i] - 0.25 and box_lats_JJA[i] + 0.25
    min_lat = box_lats_JJA[i] - 0.05
    max_lat = box_lats_JJA[i] + 0.05
    min_lon = box_lons_JJA[i] - 0.05
    max_lon = box_lons_JJA[i] + 0.05
    mask = (mat_seas['lat'] >= min_lat) & (mat_seas['lat'] <= max_lat) & (mat_seas['lon'] >= min_lon) & (mat_seas['lon'] <= max_lon)
    # get the counts at that specific window
    counts_window = mat_seas[mask].values
    # drop the first two columns (lat and lon)
    counts_window = counts_window[:, 2:]  # keep only the counts
    # sum the counts along the rows
    #counts_window = np.sum(counts_window, axis=0)
    res = stats.theilslopes(counts_window,  x=years, alpha=0.68, method='separate')
    slope = res[0]
    intercept = res[1]
    axs[i, 0].plot(years, counts_window.T, marker='o', label='Counts in the window', color=series_colors[i], markersize=5, linestyle='--')
    # plot the trend line using the slope and intercept
    axs[i, 0].plot(years, slope * years + intercept, color=series_colors[i], label='Trend line',linewidth=2)
    # add text with only 1 elements: the color and a string with the coordinates, the season, and the slope in counts/decades with specifying position
    if i < 3:
        axs[i, 0].text(0.05, 0.7, f"Box {(i+1):.0f}: +{rel_change(counts_window,slope)*100:.1f}% /10yr \n (+{slope*10:.1f} mm/h /10yr)",
                color="black", transform=axs[i, 0].transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))   
    else:    
        axs[i, 0].text(0.05, 0.85, f"Box {(i+1):.0f}: not significant",
                color="black", transform=axs[i, 0].transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))   


season = 'SON'
mat_seas = pd.read_csv(path_in + 'pctl_EXTR_eachyr_Seas_EventBased_Mean_counts_SON.txt', sep=' ')
for i in range(len(box_lons_SON)):
    plt.subplot(4, 2, 2*i + 2)
    if i == 0:
        axs[i, 1].set_title(f'SON', fontsize=20)

    axs[i, 1].set_xlim(1986, 2022)
    axs[i, 1].set_ylim(0, 53)
    axs[i, 1].grid(True, linestyle='--', alpha=0.5)
    axs[i, 0].yaxis.set_major_locator(mticker.MultipleLocator(10))
    #axs[i, 1].set_xlabel('Year', fontsize=16)
    #axs[i, 1].set_ylabel('Counts', fontsize=16)
    # create a mask for mat_seas['lat'] between box_lats_SON[i] - 0.25 and box_lats_SON[i] + 0.25
    min_lat = box_lats_SON[i] - 0.05
    max_lat = box_lats_SON[i] + 0.05
    min_lon = box_lons_SON[i] - 0.05
    max_lon = box_lons_SON[i] + 0.05
    mask = (mat_seas['lat'] >= min_lat) & (mat_seas['lat'] <= max_lat) & (mat_seas['lon'] >= min_lon) & (mat_seas['lon'] <= max_lon)
    # get the counts at that specific window
    counts_window = mat_seas[mask].values
    # drop the first two columns (lat and lon)
    counts_window = counts_window[:, 2:]  # keep only the counts
    # sum the counts along the rows
    #counts_window = np.sum(counts_window, axis=0)
    res = stats.theilslopes(counts_window,  x=years, alpha=0.68, method='separate')
    slope = res[0]
    intercept = res[1]
    axs[i, 1].plot(years, counts_window.T, marker='o', label='Counts in the window', color=series_colors[4+i], markersize=5, linestyle='--')
    # plot the trend line using the slope and intercept
    axs[i, 1].plot(years, slope * years + intercept, color=series_colors[4+i], label='Trend line',linewidth=2)
    # add text with only 1 elements: the color and a
    # string with the coordinates, the season, and the slope in counts/decades with specifying position
    if i < 3:
        axs[i, 1].text(0.05, 0.7, f"Box {(i+5):.0f}: +{rel_change(counts_window,slope)*100:.1f}% /10yr \n (+{slope*10:.1f} mm/h /10yr)",
                color="black", transform=axs[i, 1].transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))   
    else:    
        axs[i, 1].text(0.05, 0.85, f"Box {(i+5):.0f}: not significant",
                color="black", transform=axs[i, 1].transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))   

plt.tight_layout()
plt.savefig(path_out + 'pctl_EXTR_trend_single_series_8plots_rel.pdf')
