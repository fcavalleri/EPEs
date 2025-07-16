library(raster)
library(rasterVis)
library(ncdf4)
library(tidyverse)
library(cluster)
library(maps)
library(maptools)
library(RANN)
library(rcolors)
library(lattice)
library(ggpubr)
library(lubridate)

world <- map(fill = TRUE, col="transparent", plot=FALSE)
IDs <- sapply(strsplit(world$names, ":"), function(x) x[1])
world <- map2SpatialPolygons(world, IDs=IDs, proj4string=CRS("+proj=longlat +datum=WGS84"))

# input files
rean <- "MERIDA_HRES"

path_in <- "/media/kali_met_merida/METEO/Output/ope/WRFAMEZ/ECMWF/hres/"
path_out <- "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Ellipse_Example_plots/"
path_out_df <- "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Clustering/Events_Databases/"

# threshold for clustering
path_pctl <- "/media/clima_out_c1/DATI/DataForValidation/Prec/Extremes/Series/Pctl/"
r_clim_allseas <- brick(paste0(path_pctl, "MERIDA_HREShourly_wet_hours_50_pctl_seasonal_1986-2022_smooth2_25km.nc"))

# clustering parameters
thr <- 1
n_pts_min <- 5

#years of the analysis
years <- seq(1986,2022,1)

# colors for plotting
prec_col <- get_color("precip2_17lev", n = 17)[2:17]

for (year in years){

df <- data.frame(cbind(0,0,0,0,0,0,0,0,0,0,0),stringsAsFactors = TRUE)
names(df) <- c("time","lon_max","lat_max","tp_max","lon_wavg","lat_wavg",
  "area","tot_tp","eccentricity","direction","axis_maj")

dates <- seq(as.POSIXct(paste0(year,"-01-01")), as.POSIXct(paste0(year,"-12-31")), by = "hour")
yyyy_mm <- dates %>% format("%Y%m") %>% unique()
days_in_month <- c(31,28,31,30,31,30,31,31,30,31,30,31)

for (month in 1:12){

if (leap_year(year)) {days_in_month[2] <- 29} else {days_in_month[2]<-28}

infile <- paste0(path_in, yyyy_mm[month], "/PREC/", rean,"_PREC_",yyyy_mm[month],".nc")

if (month == 12){
yyyy_mm_dd_hh <- seq(as.POSIXct(paste0(year,"-",sprintf("%02d",month),"-01 01:00:00 UTC")), 
as.POSIXct(paste0(year+1,"-",sprintf("%02d",1),"-1 00:00:00 UTC")), by = "hour")
}else{
yyyy_mm_dd_hh <- seq(as.POSIXct(paste0(year,"-",sprintf("%02d",month),"-01 01:00:00 UTC")), 
as.POSIXct(paste0(year,"-",sprintf("%02d",month+1),"-1 00:00:00 UTC")), by = "hour")
}

# select the season based on the month
if (month %in% c(12,1,2)){
  season <- "DJF"
  r_clim <- r_clim_allseas[[1]]
} else if (month %in% c(3,4,5)){
  season <- "MAM"
  r_clim <- r_clim_allseas[[2]]
} else if (month %in% c(6,7,8)){
  season <- "JJA"
  r_clim <- r_clim_allseas[[3]]
} else if (month %in% c(9,10,11)){
  season <- "SON"
  r_clim <- r_clim_allseas[[4]]
}

n_hours <- length(yyyy_mm_dd_hh)

# 0) extract raster

r_all <- brick(infile)
n_levels <- dim(r_all)[3]

if (n_levels < n_hours){yyyy_mm_dd_hh <- yyyy_mm_dd_hh[1:n_levels]}

for (t in 1:length(yyyy_mm_dd_hh)){

time_str <- yyyy_mm_dd_hh[t] %>% format("%Y%m%d_%H")
time_hh <- yyyy_mm_dd_hh[t] %>% format("%Y-%m-%d %H:%M:%S UTC")
writeLines(paste0("elaborating ",time_hh))

r <- r_all[[t]]

## to see the raw field:

#png(paste0(path_out, "new_clim_test_0_",year,"_",month,"_",t,".png"), width=600, height=800)
#plot(r,asp = NULL,main="raw field",col=prec_col)
#lines(world, col="black")
#dev.off()

# 1) thresholding
r_thr <- r_norm
r_thr[r_thr < thr] <- NA

# 1.5) put original values into clusters
r_thr <- r_thr * r_clim

# 2) clustering
set.seed(0)
rc <- clump(r_thr,gaps=FALSE) 

# 3) extract general information

cluster_points <- as.data.frame(rasterToPoints(rc))
value_points <- as.data.frame(rasterToPoints(r_thr))

ind_and_value_points <- merge(cluster_points,value_points, by=c("x", "y"))
colnames(ind_and_value_points) <- c("lon", "lat", "ID", "tp")
ind_and_value_points <- ind_and_value_points[order(ind_and_value_points$ID),]

freq_cl <- freq(rc)

n_clusters <- dim(freq_cl)[1]-1

# 4) extract information for each cluster

for (i in 1:n_clusters){
  cluster_i = ind_and_value_points[ind_and_value_points[,3] == i,]

  if (dim(cluster_i)[1] > n_pts_min){
      
    tp_max_i = max(cluster_i[,4])
    lon_max_i = cluster_i[cluster_i[,4] == tp_max_i,1]
    lat_max_i = cluster_i[cluster_i[,4] == tp_max_i,2]
    lon_wavg_i = weighted.mean(cluster_i[,1], cluster_i[,4])
    lat_wavg_i = weighted.mean(cluster_i[,2], cluster_i[,4])
    area_i = freq_cl[i,2]
    tot_tp_i = sum(cluster_i[,4])

    # ellipse calculation
        exy <- try(ellipsoidhull(as.matrix(cluster_i[,1:2])))
        if (class(exy)=="ellipsoid"){
        cov_mat <- exy$cov
        eg <- eigen(cov_mat)
        axes <- sqrt(eg$values)
        direction_i <- round(atan(eg$vectors[1,1]/eg$vectors[2,1])*180.0 / pi)
        eccentricity_i <- sqrt(1-axes[2]/axes[1])
        a_i <- sqrt(exy$d2) * axes[1]  # major axis length
        } else {
        direction_i <- NA
        eccentricity_i <- NA
        a_i <- NA
        }

    df_i <- data.frame(time=time_hh, 
                                    lon_max=lon_max_i, 
                                    lat_max=lat_max_i, 
                                    tp_max=tp_max_i, 
                                    lon_wavg=lon_wavg_i, 
                                    lat_wavg=lat_wavg_i, 
                                    area=area_i, 
                                    tot_tp=tot_tp_i, 
                                    eccentricity=eccentricity_i, 
                                    direction=direction_i,
                                    axis_maj=a_i)
    df <- rbind(df, df_i)
  }
}

}

}

# eliminate firs empty row
df <- df[-1,]
# save data frame in a text file
write.table(df, paste0(path_out_df, rean,"_clusters_pctl_",year,".txt"), row.names = FALSE)

}

