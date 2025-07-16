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

# define year, month, day and t to plot
year <- 2011
month <- 10
day <- 20
hour <- 13
# t is the index of the time in the sequence of hours for the month
t <- (24*(day-1)) + hour

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

if (leap_year(year)) {days_in_month[2] <- 29} else {days_in_month[2]<-28}

infile <- paste0(path_in, yyyy_mm[month], "/PREC/", rean,"_PREC_",yyyy_mm[month],".nc")

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

time_str <- yyyy_mm_dd_hh[t] %>% format("%Y%m%d_%H")
time_hh <- yyyy_mm_dd_hh[t] %>% format("%Y-%m-%d %H:%M:%S UTC")
writeLines(paste0("elaborating ",time_hh))

r <- r_all[[t]]

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


# plot example of clustering procedure for a specific date and hour

png(paste0(path_out, "PROCEDURE_EXAMPLE_",year,"_",month,"_",t,".png"), width=2800, height=1250, res=300)
# save figure in PDF
pdf(paste0(path_out, "PROCEDURE_EXAMPLE_",year,"_",month,"_",t,".pdf"), width=10, height=4, dpi=300)

# subplot 3 in 1 row
par(mfrow=c(1,3), mar=c(2,2,2,2))
r[r < 0.001] <- NA
plot(r,asp = NULL,main=" ",col=prec_col,axes = FALSE)
mtext("a)",  side = 3, adj = 0, line = 1, cex = 1, font = 2)
yaxt = "n"
lines(world, col="black")

### To plot fields whit cluster:

cpal <- get_color("hlu_default", n = n_clusters)
r2 <- ratify(rc)
polys <- rasterToPolygons(r2, fun = function(x) !is.na(x), dissolve = TRUE)
freq_cl <- freq(rc)
n_clusters <- dim(freq_cl)[1]-1
cpal <- get_color("hlu_default", n = n_clusters)

#png(paste0(path_out, "new_clim_test_1_",year,"_",month,"_",t,".png"), width=600, height=800)
plot(r_thr,asp = NULL,main=" ",col=prec_col,axes = FALSE)
mtext("b)",  side = 3, adj = 0, line = 1, cex = 1, font = 2)
yaxt = "n"
# Plot just the contours (no fill)
plot(polys, border = cpal, col = NA, lwd = 1,add=TRUE)
lines(world, col="black")
#dev.off()

# WITH ELLIPSES

#png(paste0(path_out, "new_clim_test_ellipses_",year,"_",month,"_",t,".png"), width=600, height=800)
plot(r_thr,asp = NULL,main='',col=prec_col,axes = FALSE)
mtext("c)",  side = 3, adj = 0, line = 1, cex = 1, font = 2)
yaxt = "n"
polys <- rasterToPolygons(r2, fun = function(x) !is.na(x), dissolve = TRUE)
# Plot just the contours (no fill)
plot(polys, border = "black", col = NA, lwd = 0.5,add=TRUE)
lines(world, col="black")
n <- 0
miss_ell <- 0

for (i in 1:n_clusters){
cluster_i <- ind_and_value_points[ind_and_value_points[,3] == i,]
if (dim(cluster_i)[1] > n_pts_min){
  exy <- try(ellipsoidhull(as.matrix(cluster_i[,1:2]),tol=0.01,maxit=10000))
  if (class(exy)!="ellipsoid" || det(exy$cov)<exp(-20)){
    miss_ell<-miss_ell+1
    #writeLines(paste0("missed ",year,"_",month,"_",t," ",n))
    lon_wavg_i = weighted.mean(cluster_i[,1], cluster_i[,4])
    lat_wavg_i = weighted.mean(cluster_i[,2], cluster_i[,4])
    #text(lon_wavg_i,lat_wavg_i, paste(n), col = "green", cex = 1.3)
    }else{
    lines(predict(exy), col="red")
    #text(exy$loc[1]+0.05,exy$loc[2]-0.1, paste(n), col = 'grey23', cex = 1.4)
    }
  n <- n+1
  }
}

lines(world, col="black")
#text(18,47, paste(miss_ell,"\n missed"), col = "black", cex = 1.5)

dev.off()


