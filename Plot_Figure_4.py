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

# Define the path to the input files
df_full = pd.DataFrame()

for y in years:
    # print processed year
    print(y)
    infile = path_in + "MERIDA_HRES_clusters_pctl_" + str(y) + ".txt"
    df = pd.read_csv(infile, sep=" ")
    df_full = pd.concat([df_full, df], ignore_index=True)

# convert the "time" column to strings
df_full['time'] = df_full['time'].astype(str)

# after the "time" column, add "year", "month", "day", "hour"
# based on the "time" column formatted as YYYY-MM-DD HH:MM:SS UTC

df_full['datetime'] = pd.to_datetime(df_full['time'], format='%Y-%m-%d %H:%M:%S UTC')
#df_full['datetime'] = pd.to_datetime(df_full['time'], format='%Y%m%d')
df_full['year'] = df_full['datetime'].dt.year
df_full['month'] = df_full['datetime'].dt.month
#df_full['day'] = df_full['datetime'].dt.day
#df_full['hour'] = df_full['datetime'].dt.hour

# also add seasons based on DJF, MAM, JJA, SON
df_full['season'] = (df_full['month']%12 + 3)//3
df_full['season'] = df_full['season'].replace([1, 2, 3, 4], ['DJF', 'MAM', 'JJA', 'SON'])

# add a column "tot_tp_norm" = "tot_tp" / "area"
df_full['tot_tp_norm'] = df_full['tot_tp'] / df_full['area']
df_full['km_scale'] = df_full['axis_maj'] *110

# how much na in 'km_scale'?
print("Number of NaN in 'km_scale':", df_full['km_scale'].isna().sum())
# relative to the total number of rows
print("Relative NaN in 'km_scale':", df_full['km_scale'].isna().sum() / len(df_full))
#average area of events with km_scale == nan
#avg_tp_nan = df_full[df_full['km_scale'].isna()]['tot_tp']

# exlude where km_scale == nan
df_full = df_full.dropna(subset=['km_scale'])

# count how many rows with the same 'time'
df_full['h_counts'] = df_full.groupby('time')['time'].transform('count')
# find max counts in the 'h_counts' column
max_h_counts = df_full['h_counts'].max()
print("Max number of events per hour:", max_h_counts)
#when?
# find the time with the max number of events per hour
max_h_counts_time = df_full[df_full['h_counts'] == max_h_counts]['time'].values[0]
print("Max number of events per hour at time:", max_h_counts_time)

# histogram with different color for different seasons
df_winter = df_full[df_full['season'] == 'DJF']
df_spring = df_full[df_full['season'] == 'MAM']
df_summer = df_full[df_full['season'] == 'JJA']
df_autumn = df_full[df_full['season'] == 'SON']



# seasonal histograms (Figure 4)
vars_to_hist = ["h_counts"]
udm = ["events/h"]
names = ["Number of events per hour"]
# set breaks as a vector with 0 and after np.arange(1, 141, 5)
breaks = np.arange(1, 100, 5)

figure = plt.figure(figsize=(6, 4))
plt.xlim(1, 100)
#DJF
n_events_per_h = df_winter.groupby('datetime').size().reset_index(name='h_counts')
# concatenate to n_events_per_h a vector of (24*90* 37) - len(n_events_per_h) zeros to the end of the array
arr_hist , edges = np.histogram(n_events_per_h['h_counts'], bins=breaks)

arr_norm = arr_hist / (24*90* 37)  # 24 hours * 90 days * 37 years
plt.plot(edges[0:-1] , arr_norm, 'blue',linestyle='--', linewidth=1.5, label='JJA')

#MAM
n_events_per_h = df_spring.groupby('datetime').size().reset_index(name='h_counts')
#n_events_per_h = n_events_per_h.set_index('datetime').reindex(pd.date_range(start='1986-01-01 01:00', end='2022-12-31 23:00', freq='H'), fill_value=0).reset_index()
arr_hist , edges = np.histogram(n_events_per_h['h_counts'], bins=breaks)
arr_norm = arr_hist / (24*92* 37)  # 24 hours * 92 days * 37 years
plt.plot(edges[0:-1] , arr_norm, 'green',linestyle='--', linewidth=1.5, label='MAM')
#JJA
n_events_per_h = df_summer.groupby('datetime').size().reset_index(name='h_counts')
#n_events_per_h = n_events_per_h.set_index('datetime').reindex(pd.date_range(start='1986-01-01 01:00', end='2022-12-31 23:00', freq='H'), fill_value=0).reset_index()
arr_hist , edges = np.histogram(n_events_per_h['h_counts'], bins=breaks)
arr_norm = arr_hist / (24*92* 37)  # 24 hours * 92 days * 37 years
plt.plot(edges[0:-1] , arr_norm, 'red',linestyle='--', linewidth=1.5, label='JJA')
#SON
n_events_per_h = df_autumn.groupby('datetime').size().reset_index(name='h_counts')
#n_events_per_h = n_events_per_h.set_index('datetime').reindex(pd.date_range(start='1986-01-01 01:00', end='2022-12-31 23:00', freq='H'), fill_value=0).reset_index()
arr_hist , edges = np.histogram(n_events_per_h['h_counts'], bins=breaks)
arr_norm = arr_hist / (24*91* 37)  # 24 hours * 91 days * 37 years
plt.plot(edges[0:-1] , arr_norm, 'orange',linestyle='--', linewidth=1.5, label='SON')

plt.xlabel(udm[0])
plt.yscale('linear')
plt.ylabel("probability density")
# specify xticks 1, 5, 10, 15, ...
plt.xticks(np.concatenate(([1], np.arange(10, 100, 10))))
#plt.ylim(10e-7, 1)
plt.grid()
plt.legend(loc='upper center',labels=['DJF', 'MAM', 'JJA', 'SON'], title='Seasons', fontsize=10)

plt.savefig(path_out + 'Figure_4.pdf')


# how much dry hours in each season?
perc_dry_h = (1 - df_winter['time'].nunique() / (24*90* 37)) *100 # 24 hours * 90 days * 37 years
print(f"dry DJF: {perc_dry_h:.2f}%")
perc_dry_h = (1 - df_spring['time'].nunique() / (24*92* 37))  *100# 24 hours * 92 days * 37 years
print(f"dry MAM: {perc_dry_h:.2f}%")
perc_dry_h = (1 - df_summer['time'].nunique() / (24*92* 37))  *100# 24 hours * 92 days * 37 years
print(f"dry JJA: {perc_dry_h:.2f}%")
perc_dry_h = (1 - df_autumn['time'].nunique() / (24*91* 37)) *100 # 24 hours * 91 days * 37 years
print(f"dry SON: {perc_dry_h:.2f}%")