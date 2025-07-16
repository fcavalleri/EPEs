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
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatterSciNotation

rean = "MERIDA-HRES"

path_out = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/Plots/"
path_out_npv = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/AveragingVals/"
path_in = "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/"

years = np.linspace(1986, 2022, 37)  # 1986-2020 for MOLOCH, 1986-2022 for MERIDA-HRES
years = years.astype(int)

## Load extreme dataset
df_extreme = pd.read_csv(path_in + rean + '_pctl_extremes_1986-2022.txt', sep=' ')
df_full=df_extreme
df_full['km_scale'] = df_full['axis_maj'] *110
df_full = df_full.dropna(subset=['km_scale'])

########################
# histograms:
vars_to_hist = ["tot_tp_norm","tp_max","km_scale"]
udm = ["mm/h", "mm/h", "km"]
names = ["EPEs tot_tp/area", "EPEs tp_max", "EPEs axis_maj"]
lett = ["a)", "b)", "c)"]

# total number of events
tot_events = df_full['tot_tp'].count()

# histogram with different color for different seasons
df_winter = df_full[df_full['season'] == 'DJF']
df_spring = df_full[df_full['season'] == 'MAM']
df_summer = df_full[df_full['season'] == 'JJA']
df_autumn = df_full[df_full['season'] == 'SON']

# put seasonal plots for all vars_to_hist in the same figure
# define xlimits for each variable

breaks = {
"tot_tp_norm": np.arange(0, 100, 0.5),
"tp_max": np.arange(0, 400, 0.5),
"km_scale": np.arange(0, 1000, 2)
}

xlimits = {
    "tot_tp_norm": (0, 25),
    "tp_max": (0, 75),
    "km_scale": (0, 175)
}

ylimits = {
    "tot_tp_norm": (0, 0.04),
    "tp_max": (0, 0.015),
    "km_scale": (0, 0.03)    
}

figure = plt.figure(figsize=(10, 4))
#plt.suptitle('Seasonal histograms of variables', fontsize=20)

for i, var in enumerate(vars_to_hist):
    plt.subplot(1, 3, i+1)
    #DJF
    arr_hist , edges = np.histogram(df_winter[var], bins = breaks[var])
    arr_norm = arr_hist / tot_events
    plt.plot(edges[0:-1] , arr_norm, 'blue',linestyle='--', linewidth=1.5, label='DJF')
    # print the values of edge steps
    print(f"Edges for {var} in DJF: {np.diff(edges)[1]}")    
    #MAM
    arr_hist , edges = np.histogram(df_spring[var], bins =  breaks[var])
    arr_norm = arr_hist / tot_events
    plt.plot(edges[0:-1] , arr_norm, 'green',linestyle='--', linewidth=1.5, label='MAM')
    print(f"Edges for {var} in MAM: {np.diff(edges)[1]}")
    #JJA
    arr_hist , edges = np.histogram(df_summer[var], bins =  breaks[var])
    arr_norm = arr_hist / tot_events
    plt.plot(edges[0:-1] , arr_norm, 'red',linestyle='--', linewidth=1.5, label='JJA')
    print(f"Edges for {var} in JJA: {np.diff(edges)[1]}")
    #SON
    arr_hist , edges = np.histogram(df_autumn[var], bins =   breaks[var])
    arr_norm = arr_hist / tot_events
    plt.plot(edges[0:-1] , arr_norm, 'orange',linestyle='--', linewidth=1.5, label='SON')
    print(f"Edges for {var} in SON: {np.diff(edges)[1]}")
    # #plt.hist(df_winter[var], ec='blue', fc='none', lw=1.5, histtype='step', label='DJF',density="False",bins=50)
    #plt.hist(df_spring[var], ec='green', fc='none', lw=1.5, histtype='step', label='MAM',density="False",bins=50)
    #plt.hist(df_summer[var], ec='red', fc='none', lw=1.5, histtype='step', label='JJA',density="False",bins=50)
    #plt.hist(df_autumn[var], ec='orange', fc='none', lw=1.5, histtype='step', label='SON',density="False",bins=50)
    plt.xlabel(names[i] + ' (' + udm[vars_to_hist.index(var)] + ')')
    plt.yscale('linear')
    plt.ylabel("probability density")
    plt.ylim(ylimits[var])
    # get xlim from xlimits dictionary
    plt.xlim(xlimits[var])
    #plt.ylim(0, 0.15)
    plt.grid()
    #
    if i == 0:
        plt.legend(loc='upper right',labels=['DJF', 'MAM', 'JJA', 'SON'], title='Seasons', fontsize=10)
    plt.title(lett[i], fontsize=15, loc='left')
# add common seasonal legend
plt.tight_layout()
plt.savefig(path_out + 'Figure_10.pdf')
plt.close()

# percentual of events outside the histogram x limits
for var in vars_to_hist:
    xlim = xlimits[var]
    # count how many events are outside the xlim
    outside = df_full[(df_full[var] < xlim[0]) | (df_full[var] > xlim[1])]
    perc_outside = len(outside) / len(df_full) * 100
    print(f"Percentage of events outside the histogram limits for {var}: {perc_outside:.2f}%")
