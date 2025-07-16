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
window_size = 0.5  
step_size = 0.1  

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

#seasons = ['DJF', 'MAM', 'JJA', 'SON']
seasons = ['JJA', 'SON']

for var in vars_to_trend:

    for season in seasons:
        # print var and season
        print(var, season)
        #load matrix saved as txt at df_counts_avg.to_csv(path_out + 'EXT_eachyr_Seas_EventBased_Mean_counts_' + seas + '.txt', sep=' ', index=False)
        mat_seas = pd.read_csv(path_in + 'pctl_EXTR_eachyr_Seas_EventBased_Mean_' + var + '_' + season + '.txt', sep=' ')
        # how much Nan for each column
        #print(mat_seas.isna().sum())
        
        # apply stats.theilslopes to each row excluding lon and lat
        # vector to store slopes
        slopes = np.zeros(len(mat_seas))
        pvals = np.zeros(len(mat_seas))
        #zero_years = np.zeros(len(mat_seas))
        #avg_year = np.zeros(len(mat_seas))

        for i in range(len(mat_seas)):
            #print(i)
            # if mat_seas.iloc[i, 2:] has more than 10 NaN values, put NaN in slopes[i]
            #zero_years[i] = np.sum((mat_seas.iloc[i, 2:]==0))
            #find indices of values different from 0
            #nozero_indices = np.where(mat_seas.iloc[i, 2:] != 0)[0]
            # get the years corresponding to the nozero_indices
            #zero_years = years[nozero_indices]
            #avg_year[i] = np.mean(zero_years)

        #np.savetxt(path_in + "pctl_EXTR_avg_year_" + season + '.txt', zero_years, delimiter=' ')

            if np.sum(np.isnan(mat_seas.iloc[i, 2:])) > 10:
                slopes[i] = np.nan
                pvals[i] = 1
            else:
                res = stats.theilslopes(mat_seas.iloc[i, 2:],  x=years, alpha=0.68, method='separate')
                pvals[i] = mk.original_test(mat_seas.iloc[i, 2:]).p
                slopes[i] = res[0]

        # normalization
        clim_val = np.nanmean(mat_seas.iloc[:, 2:], axis=1)
        slopes_norm = slopes *10 / clim_val 

        # adjusted p-value
        rejected,adj_pvals = statsmodels.fdrcorrection(pvals, alpha=0.05, method='i', is_sorted=False)

        # put 0 values to the slopes_norm where adj_pvals > 0.1
        slopes_norm_masked = np.where(adj_pvals > 0.1, 0, slopes_norm)
        slopes_masked = np.where(adj_pvals > 0.1, 0, slopes)
        signif_mask = np.where(adj_pvals > 0.1, 0, 1)
        # put signif_mask to 0 where slopes_norm_masked is between -0.1 and 0.1
        signif_mask = np.where((slopes_norm_masked < 0.1) & (slopes_norm_masked > -0.1), 0, signif_mask)
        # save the slopes_norm_masked as txt 
        np.savetxt(path_in + "pctl_EXTR_" + var + '_slopes_norm_' + season + '.txt', slopes_norm, delimiter=' ')
        np.savetxt(path_in + "pctl_EXTR_" + var + '_signif_mask_' + season + '.txt', signif_mask, delimiter=' ')
        #np.savetxt(path_in + "pctl_EXTR_" + var + '_zero_years_' + season + '.txt', zero_years, delimiter=' ')

