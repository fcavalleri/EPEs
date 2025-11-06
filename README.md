### EPEs
This repository contains the code used for an event-based analysis of hourly precipitation across Italy, with a focus on the potential intensification of extreme events. 
The study underlying this work is currently under peer review. The derived event dataset is openly available on Zenodo: https://bit.ly/HOPSS-X.

# Event-detection algorithm (in R):
- Event_detection.R
  Takes hourly precipitation fields from MERIDA HRES reanalysis as input,
  provide the MERIDA_HRES_clusters_pctl_YEAR.txt files with the events.

# Statistical analyses (in Python):
- Moving_window_averages.py
  Takes MERIDA_HRES_clusters_pctl_YEAR.txt files as inputs,
  provides the averaged values with moving window as:
  pctl_eachyr_Seas_EventBased_Mean_VAR_SEAS.txt
  for SEAS in DJF, MAM, JJA, SON and VAR in N, AvIn, PkIn, SpS
- EPEs_subsetting.py
  Takes MERIDA_HRES_clusters_pctl_YEAR.txt files as inputs,
  Uses the file Annual_extremes_MERIDA_HRES_1h_max_clim_1986-2022_2smooth20km.nc as threshold,
  provides the file MERIDA-HRES_pctl_extremes_1986-2022.txt with the EPEs
- EPEs_Moving_window_averages.py
  Takes MERIDA-HRES_pctl_extremes_1986-2022.txt  file as input,
  provides the averaged values with moving window as:
  pctl_EXTR_eachyr_Seas_EventBased_Mean_VAR_SEAS.txt
  for SEAS in DJF, MAM, JJA, SON and VAR in N, AvIn, PkIn, SpS
- EPEs_trend.py
  Takes pctl_EXTR_eachyr_Seas_EventBased_Mean_VAR_SEAS.txt files as inputs,
  provides pctl_EXTR_VAR_slopes_norm_SEAS.txt (trends)
  and pctl_EXTR_VAR_signif_mask_SEAS.txt (significance)
  for SEAS in DJF, MAM, JJA, SON and VAR in N, AvIn, PkIn, SpS
- EPEs_persitence.py
  Takes MERIDA-HRES_pctl_extremes_1986-2022.txt  file as input,
  provides pctl_EXTR_clusters_mean_1986-2022_SEAS.txt (persistence)

# Figures plotting (in Phyton):
