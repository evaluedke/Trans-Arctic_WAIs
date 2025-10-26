# hallo hier sind alle meine funktionen drin für die Lifecycle Analyse
# author: eva luedke 
# 2025

###############################################################
######################IMPORT PACKAGES##########################
###############################################################
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.path as mpath
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import cartopy.feature
import numpy as np
import pandas as pd
import scipy
import datetime
import cartopy
from metpy.units import units
import cartopy.crs as ccrs
import pickle
import os
import glob
import subprocess
import dask
from joblib import Parallel, delayed
from geopy.distance import distance
from enum import Enum
import concurrent.futures
import pyproj
from pyproj import Geod

### Load MOAAP functions
import sys
sys.path.append('/work/aa0049/a271122/MOAAP/MOAAP/')
from src.utils import *
from src.Corrections import * 
from src.Enumerations import Month, Season, Experiments, Domains
from src.xarray_util import create_obj_from_dict,  ObjectContainer,  load_tracking_objects
from src.plot_funcs import plot_contourf_rotated_grid #, plot_unstructured_rotated_grid
from src.GridPoints import Domain

###Load Domain Definitions
class Domains(Enum):
    # JLa custom domains
    NORTH_ATLANTIC_JLa = Domain(north=80, south=67, east=60, west=-30)
    BEAUFORT_SIBERIAN_JLa = Domain(north=80, south=70, east=-120, west=120)
    CENTRAL_ARCTIC_JLa = Domain(north=90, south=80, east=180, west=-180)



###Define stuff
ds=xr.open_dataset(
    '/work/aa0049/a271041/spice-v2.1/chain/work/run_era5_polarres_wp3_hindcast2/post/yearly/T_2M/T_2M_2020010100-2020123123.ncz')

# extract the rotated coordinate system
crs_rot = ccrs.RotatedPole(pole_longitude=ds.rotated_pole.attrs['grid_north_pole_longitude'],\
                           pole_latitude=ds.rotated_pole.attrs['grid_north_pole_latitude'])

#Domains
gridspacing=0.75
lat=np.arange(-90, 90+gridspacing,gridspacing)
lon=np.arange(-180, 180+gridspacing,gridspacing)
ds_dummy = xr.Dataset({"dummy": (("lat", "lon"), np.ones([len(lat), len(lon)]))},coords={"lon": lon,"lat": lat})

dd=ds_dummy.copy()
mask_domain_CentralArctic=dd.dummy.where(dd.lat>80).notnull()
mask_domain_NorthAtlantic=dd.dummy.where((dd.lat<=80) & (dd.lat>67) & (dd.lon>-30) & (dd.lon<60)).notnull()
mask_domain_BeaufortSiberianSea1=dd.dummy.where((dd.lat<=80) & (dd.lat>70) & (dd.lon>120)).notnull()
mask_domain_BeaufortSiberianSea2=dd.dummy.where((dd.lat<=80) & (dd.lat>70) & (dd.lon<-120)).notnull()

mask_domain_BeaufortSiberianSea = mask_domain_BeaufortSiberianSea1 | mask_domain_BeaufortSiberianSea2


mask_domains = {
    "CentralArctic": {"mask": mask_domain_CentralArctic, "color": 'darkred',"north":90,"south":80,"east":180,"west":-180},
    "NorthAtlantic": {"mask": mask_domain_NorthAtlantic, "color": 'darkblue',"north":80,"south":67,"east":60,"west":-30},
    "BeaufortSiberianSea": {"mask": mask_domain_BeaufortSiberianSea, "color": "gold","north":80,"south":70,"east":120,"west":-120},
}
domains = ["NorthAtlantic","CentralArctic","BeaufortSiberianSea"]


###############################################################
#########################FUNCTIONS#############################
###############################################################

### MOAAP Select and Filter Objects 
def define_df_objs(objs):
    '''Function to create dataframe that contains MOAAP IVT Object IDS and color'''
    df_objs = pd.DataFrame({
        'id': [ds.id_.item() for ds in objs],
        'color': [plt.get_cmap("tab20b")(i / len(objs))[:3] for i in range(len(objs))]
    })
    return df_objs




def sel_objs(objs,season):
    '''Function to select MOAAP IVT Objects that pass the three domains NA, CA, BS'''

    objs_sel = objs.sel_season(Season[season]) \
                                .sel_by_domain(Domains.NORTH_ATLANTIC_JLa.value,
                                               type_="anytime",domain_frac=0.,select_last_timesteps = True) \
                                .sel_by_domain(Domains.CENTRAL_ARCTIC_JLa.value,type_="anytime",
                                               domain_frac=0.,select_last_timesteps = True)  \
                                .sel_by_domain(Domains.BEAUFORT_SIBERIAN_JLa.value,type_="anytime",
                                               domain_frac=0.,select_last_timesteps= True)

    
    ids_sel=[ds.id_.item() for ds in objs_sel]

    return objs_sel, ids_sel

def extract_and_transform_rot_coords(objs_):
    # Extract rotated coordinates 
    for nr in range(len(objs_)):
        
        latitudes=[]
        longitudes=[]
        for i in range(len(objs_[nr].track)):
            latitudes.append(objs_[nr].track[i].item().lat)
            longitudes.append(objs_[nr].track[i].item().lon)
        
        lat_obj=xr.DataArray(latitudes,dims=["times"],coords={"times":objs_[nr]["times"]})
        lon_obj=xr.DataArray(longitudes,dims=["times"],coords={"times":objs_[nr]["times"]})
        
        objs_[nr]["lat"]=lat_obj
        objs_[nr]["lon"]=lon_obj
        
    #For all tracks in objs_sel, transform the rotated coordinates into standard (ccrs.PlateCarree() coordinates and store them in additional data variables geo_lat and geo_lon
    for nr in range(len(objs_)): 
        standard_crs = ccrs.PlateCarree()
    
        rotated_lons = objs_[nr]['lon'].values
        rotated_lats = objs_[nr]['lat'].values
        
        transformed_coords = standard_crs.transform_points(crs_rot, rotated_lons, rotated_lats)
    
        geo_lat=xr.DataArray(transformed_coords[...,1] ,dims=["times"],coords={"times":objs_[nr]["times"]}) #transformed latitudes
        geo_lon=xr.DataArray(transformed_coords[...,0],dims=["times"],coords={"times":objs_[nr]["times"]}) #transformed longitudes
        
        objs_[nr]["geo_lat"]=geo_lat
        objs_[nr]["geo_lon"]=geo_lon

    return objs_ 


def extract_domain_times(ds_obj,north_boundary,south_boundary,east_boundary,west_boundary,BS=False):
    
    #Domain boundaries definition
    north_boundary = north_boundary
    south_boundary = south_boundary
    east_boundary = east_boundary
    west_boundary = west_boundary
    
    timesList = []
    for time in ds_obj['times']:
    
        lat = ds_obj.geo_lat.sel(times=time.values).item()
        lon = ds_obj.geo_lon.sel(times=time.values).item()

        if BS == True: 
            if (70 <= lat <= 80) and ((120 <= lon <= 180) or (-180 <=lon <= -120)):
                timesList.append(time.values) 
        else: 
        #Check if gridpoint is in Domain 
            if (south_boundary <= lat <= north_boundary) and (west_boundary <= lon <= east_boundary):
                timesList.append(time.values)
    return timesList


def combine_times(objs_):
    '''Returns lists that contain all times in which IVT Objects are in NA, CA, BS domain'''
    #North_Atlantic
    NA_times = []
    for nr in range(len(objs_)):
        NA_time=extract_domain_times(objs_[nr],80,67,60,-30)
        NA_times.append(NA_time)
    
    NA_combined_times = np.concatenate(NA_times)
    
    #Central_Arctic
    CA_times = []
    for nr in range(len(objs_)):
        CA_time=extract_domain_times(objs_[nr],90,80,180,-180)
        CA_times.append(CA_time)
    
    CA_combined_times = np.concatenate(CA_times)
    
    #Beaufort_Siberian
    BS_times = []
    for nr in range(len(objs_)):
        BS_time=extract_domain_times(objs_[nr],80,70,120,-120,True)
        BS_times.append(BS_time)
    
    BS_combined_times = np.concatenate(BS_times)

    return NA_combined_times, CA_combined_times, BS_combined_times

def create_final_objs(objs_):
    '''Filter IVT Objects that come from the Atlantic'''
    objs_final = []
    for nr in range(len(objs_)):
        NA_time=extract_domain_times(objs_[nr],80,67,60,-30)
        BS_time=extract_domain_times(objs_[nr],80,70,120,-120,True)
        CA_time=extract_domain_times(objs_[nr],90,80,180,-180)
    
        if NA_time[0] < BS_time[0] and NA_time[0] < CA_time[0] and CA_time[0] < BS_time[0]:
            objs_final.append(objs_[nr])

    
    print(len(objs_final))
    return objs_final 

### Functions to plot composites

def get_domain_values(ds,clim_DJF,NA_combined_times,CA_combined_times,BS_combined_times):
    '''Returns 2D mean values of variable during time in which tracks exist in domain 
    NorthAtlantic, CentralArctic, and BeaufortSiberianSea
    Returns climatology of the calendar days in which tracks exist in domain'''

    var_NA = ds.sel(time=NA_combined_times, method='nearest').mean(dim='time')
    var_CA = ds.sel(time=CA_combined_times, method='nearest').mean(dim='time')
    var_BS = ds.sel(time=BS_combined_times, method='nearest').mean(dim='time')
    
    #Climatology
    
    #Get calendar days of domain times
    NA_month_day = list(set((time.month, time.day) for time in pd.to_datetime(NA_combined_times)))
    CA_month_day = list(set((time.month, time.day) for time in pd.to_datetime(CA_combined_times)))
    BS_month_day = list(set((time.month, time.day) for time in pd.to_datetime(BS_combined_times)))
    
    clim_NA = clim_DJF.where(
        clim_DJF['time'].dt.strftime('%m-%d').isin([f"{month:02d}-{day:02d}" for month,day in NA_month_day]), drop=True)
    clim_CA = clim_DJF.where(
        clim_DJF['time'].dt.strftime('%m-%d').isin([f"{month:02d}-{day:02d}" for month,day in CA_month_day]), drop=True)
    clim_BS = clim_DJF.where(
        clim_DJF['time'].dt.strftime('%m-%d').isin([f"{month:02d}-{day:02d}" for month,day in BS_month_day]), drop=True)
        
    return var_NA, var_CA, var_BS, clim_NA, clim_CA, clim_BS 


def get_month_day(objs_time):
    #Extract month, day pairs from track times
    
    track_times_pd = pd.to_datetime(objs_time.values)
    month_day = list(set((time.month, time.day) for time in track_times_pd))
    return month_day


def plot_domain_mean(ds_domain_mean,ds_contour_mean,axs,domain,title,label_cbar,objs_,cmap,sic,vmin=None,vmax=None):
    '''Function to plot single variable, sea ice edge, and domain mask'''
    
    #plot variable
    variable =  ds_domain_mean.plot(ax=axs,transform=crs_rot,cmap=cmap,vmin=vmin,vmax=vmax, add_colorbar = False)
   
    # Add custom colorbar
    colorbar = plt.colorbar(variable, ax=axs, orientation='vertical', pad=0.06, shrink=0.8)
    colorbar.set_label(label_cbar, fontsize=10)
    colorbar.ax.tick_params(labelsize=10)
    
    #Add contour
    #contour = ds_contour_mean.plot.contour(levels=11,ax=axs,transform=crs_rot,colors='k',linewidths=0.5)
    #axs.clabel(contour, fmt='%1.0f', inline=True, fontsize=7)

    #Add sea ice
    (sic>85).plot.contour(levels=1, ax = axs, transform = crs_rot, colors = 'cyan', linewidths = 1.5,
                                   linestyles = '-',zorder=2)
    
    #Add domain mask
    mask_domains[domain]["mask"].plot.contour(levels=1,colors=mask_domains[domain]["color"],\
        linewidths=2,linestyles='--',zorder=4,ax=axs,transform=ccrs.PlateCarree())

    axs.set_title(f"{title} sector",y=1.05)


#Select colormap, diverging if anomaly and sequential for total values
def plot_all_domain_means(NA_vals,CA_vals,BS_vals,NA_contour,BS_contour,CA_contour,label_cbar,objs_,sic,vmin,vmax):
    '''Function to plot composites of all three domains'''
    cmap = 'RdBu_r'
    fig, axs = plt.subplots(1,3,subplot_kw={'projection': ccrs.Orthographic(0,90)},figsize=(16,4))
    
    #For 3 selected domains, create composite plot including variable mean, contour lines
    plot_domain_mean(NA_vals,NA_contour,axs[0],"NorthAtlantic","North Atlantic",label_cbar,objs_,cmap,sic,vmin,vmax)
    plot_domain_mean(CA_vals,CA_contour,axs[1],"CentralArctic","Central Arctic",label_cbar,objs_,cmap,sic,vmin,vmax)
    plot_domain_mean(BS_vals,BS_contour,axs[2],"BeaufortSiberianSea","Beaufort/Siberian Sea",label_cbar,objs_,cmap,sic,vmin,vmax)

    #Set extent, gridlines, circular shape
    for aa in axs:
        
        aa.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
        #aa.gridlines(linestyle='--',zorder=3)
        aa.coastlines(zorder=2)
        aa.add_feature(cartopy.feature.LAND, color='lightgray',zorder=0, edgecolor='black') 
        aa.add_feature(cartopy.feature.OCEAN, color='white',zorder=0, edgecolor='black')

        aa.set_aspect('equal')
        center, radius = [0.5, 0.5], 0.55
        circle = mpath.Path(np.vstack([np.sin(np.linspace(0, 2*np.pi, 100)), np.cos(np.linspace(0, 2*np.pi, 100))]).T * radius + center)
        aa.set_boundary(circle, transform=aa.transAxes)


def create_composites(
    var,varname,
    NA_combined_times,CA_combined_times,BS_combined_times,
    psl_NA_anom,psl_CA_anom,psl_BS_anom,
    ICON_path,ICON_clim_path,
    objs_final,
    sic,
    vmin=None,vmax=None,ssp=False,period='',
    startdate='',enddate='',
    NESM=False):
    
    '''Function to load timeseries dataset and DJF climatology, compute anomalies and plot them for 3 domains'''

    #Load dataset and climatology
    if NESM: 
        ds = xr.open_mfdataset(f'{ICON_path}{var}/v20250601/{var}_*.nc',chunks = {'time':10})
    else: 
        ds = xr.open_mfdataset(f'{ICON_path}{var}/{var}_*.nc',chunks = {'time':10})
    if ssp: 
        ds_clim = xr.open_dataset(f'{ICON_clim_path}/clim/{var}/{var}_clim_day_DJF_ssp_{period}.nc',chunks={'time': 10})
        ds = ds.sel(time=slice(startdate, enddate))
    else: 
        ds_clim = xr.open_dataset(f'{ICON_clim_path}/clim/{var}/{var}_clim_day_DJF.nc',chunks={'time': 10})

    #Select dataset and climatology for different domain times
    var_NA, var_CA, var_BS, var_clim_NA, var_clim_CA, var_clim_BS = get_domain_values(ds[varname],ds_clim,
                                                                        NA_combined_times,CA_combined_times,BS_combined_times)

    #Compute anomalies
    var_NA_anom = var_NA - var_clim_NA[varname].mean(dim='time')
    var_CA_anom = var_CA - var_clim_CA[varname].mean(dim='time')
    var_BS_anom = var_BS - var_clim_BS[varname].mean(dim='time')

    objs_final = objs_final

    #Plot Composite for 3 domains
    plot_all_domain_means(var_NA_anom,var_CA_anom,var_BS_anom,psl_NA_anom,psl_CA_anom,psl_BS_anom,
                          f'{ds[varname].standard_name} [{ds[varname].units}]',objs_final,sic,vmin,vmax)




### Lifecycle Analysis 

def create_masked_lists_shapes(var_data, domains,objs_final,ds_moaap):
    '''
    Function to save masked and area weighted data points, returns them in list_shape
    Definition of domain times through shapes in domain. 
    Minimum number of points to get a valid value need to be defined through min_valid_points
    '''
    list_shapes = {domain: [] for domain in domains}
    min_valid_points = 10 
    
    for i in range(len(objs_final)):

        time_i=objs_final[i].times
        
        #Load MOAAP shapes for all event timesteps
        moaap_sel = ds_moaap.sel(time=time_i, method='nearest')
        mask = (moaap_sel.IVT_Objects == int(objs_final[i].id_.item()))
    
        #Load variable data for the same timesteps
        data_sel = var_data.sel(time=time_i, method='nearest')
    
        # Align mask with data grid
        aligned_mask = mask.astype(int).interp(rlat=data_sel.rlat, rlon=data_sel.rlon, method="nearest")
        aligned_mask = aligned_mask > 0
    
        # Apply the mask on the variable data
        filtered_data = data_sel.where(aligned_mask)
        filtered_data = filtered_data.assign_coords(lon=filtered_data.lon)
        
        for domain in domains:
            if domain == "BeaufortSiberianSea":
                filtered_domain = filtered_data.where(
                    (
                        (filtered_data.lat >= mask_domains[domain]["south"]) &
                        (filtered_data.lat <= mask_domains[domain]["north"]) &
                        (filtered_data.lon >= mask_domains[domain]["east"]) &
                        (filtered_data.lon <= 180)
                    ) |
                    (
                        (filtered_data.lat >= mask_domains[domain]["south"]) &
                        (filtered_data.lat <= mask_domains[domain]["north"]) &
                        (filtered_data.lon >= -180) &
                        (filtered_data.lon <= mask_domains[domain]["west"])
                    )
                )
            else:
                # Apply standard domain mask
                filtered_domain = filtered_data.where(
                    (filtered_data.lat >= mask_domains[domain]["south"]) &
                    (filtered_data.lat <= mask_domains[domain]["north"]) &
                    (filtered_data.lon >= mask_domains[domain]["west"]) &
                    (filtered_data.lon <= mask_domains[domain]["east"])
                )
                
            #Check if filtered_domain has at least min_valid_points
            num_valid_points = filtered_domain.count()
        
            if num_valid_points >= min_valid_points:
        
                #Create an area weighted mean using test_weights and add mean value to the list
                filtered_mean = filtered_domain.mean(("rlat","rlon"),skipna=True)

                #Remove NaN values
                valid_filtered_mean = filtered_mean.dropna(dim='times', how='all')
         
                list_shapes[domain].append(valid_filtered_mean)

    # Return lists for all domains
    return [list_shapes[domain] for domain in domains]

###Anomalies

def create_masked_lists_anomalies(var_data,clim,domains,objs_final,ds_moaap): 
    '''Function to save masked and area weighted data points in a list, save id'''
    list_shapes = {domain: [] for domain in domains}
    min_valid_points = 10 

    for i in range(len(objs_final)):
        objs_plot_ids = objs_final[i]['id_'].item()

        time_i=objs_final[i].times
        
        #Select climatology for specific calendar days
        month_day = list(set((time.month, time.day) for time in pd.to_datetime(time_i)))
        clim_daysofyear = clim.where(clim['time'].dt.strftime('%m-%d').isin([f"{month:02d}-{day:02d}" for month,
                                                                             day in month_day]), drop=True)
                
        moaap_sel = ds_moaap.sel(time=time_i,method='nearest')
        data_sel = var_data.sel(time=time_i,method='nearest') - clim_daysofyear.mean(dim='time')
    
        #Create mask of chosen object and align with data grid
        mask = (moaap_sel.IVT_Objects==int(objs_plot_ids))
        aligned_mask = mask.astype(int).interp(rlat=data_sel.rlat, rlon=data_sel.rlon, method="nearest")
        aligned_mask = aligned_mask > 0
        
        #Apply the mask to the data
        filtered_data = data_sel.where(aligned_mask)
        filtered_data = filtered_data.assign_coords(lon=filtered_data.lon)   

        for domain in domains:
            # Apply domain mask
            if domain == "BeaufortSiberianSea":
                filtered_domain = filtered_data.where(
                    (
                        (filtered_data.lat >= mask_domains[domain]["south"]) &
                        (filtered_data.lat <= mask_domains[domain]["north"]) &
                        (filtered_data.lon >= mask_domains[domain]["east"]) &
                        (filtered_data.lon <= 180)
                    ) |
                    (
                        (filtered_data.lat >= mask_domains[domain]["south"]) &
                        (filtered_data.lat <= mask_domains[domain]["north"]) &
                        (filtered_data.lon >= -180) &
                        (filtered_data.lon <= mask_domains[domain]["west"])
                    )
                )
                
            else:
                filtered_domain = filtered_data.where(
                    (filtered_data.lat >= mask_domains[domain]["south"]) &
                    (filtered_data.lat <= mask_domains[domain]["north"]) &
                    (filtered_data.lon >= mask_domains[domain]["west"]) &
                    (filtered_data.lon <= mask_domains[domain]["east"])
                )

            
            #Check if filtered_domain has at least min_valid_points
            num_valid_points = filtered_domain.count()
        
            if num_valid_points >= min_valid_points:
        
                #Create an area weighted mean using test_weights and add mean value to the list
                filtered_mean = filtered_domain.mean(("rlat","rlon"),skipna=True)
    
                #Remove NaN values
                valid_filtered_mean = filtered_mean.dropna(dim='times', how='all')
         
                list_shapes[domain].append(valid_filtered_mean)

    # Return lists for all domains
    return [list_shapes[domain] for domain in domains]


def plot_scatter_domains(NA_shapes, CA_shapes, BS_shapes, list_ids, ylabel, title, savefig = False):
    '''Function to create a scatter plot for each shape mean value per timestep for all objects together with the median value of all 
    shapes of one object'''
    #Colormap
    cmap = cm.get_cmap("tab20b", len(list_ids))
    color_map = {idx: cmap(i) for i, idx in enumerate(list_ids)}
    
    all_shapes = [(NA_shapes, 1), (CA_shapes, 2), (BS_shapes, 3)]
    jitter = 0.2
    
    plt.figure(figsize=(7, 6))
    
    # Plot all shapes at 3 positions with shapemean
    for shapes, x_position in all_shapes:
        for shape, id_ in zip(shapes, list_ids):
            num_points = len(shape)
            x_coords = np.full(num_points, x_position) + np.random.uniform(-jitter, jitter, num_points)
            shape_mean = np.nanmedian(shape.values)
            #print(np.nanmean(shape.values))

            #Differenciate between 1st and other positions to prevent legend from showing duplicate values 
            if x_position == 1:
                plt.scatter(x=x_coords, y=shape, color=color_map[id_],alpha=0.2,zorder=1)
                plt.scatter(x=x_position, y = shape_mean, label=id_,color = color_map[id_],alpha=1,zorder=2,edgecolors= 'k')
              
            else:
                plt.scatter(x=x_coords, y=shape, color=color_map[id_],alpha=0.2)
                plt.scatter(x=x_position, y = shape_mean, color = color_map[id_],alpha=1,zorder=2,edgecolors= 'k')
    
    # Plot settings
    plt.xticks(ticks=[1, 2, 3], labels=['North Atlantic', 'Central Arctic', 'Beaufort Sea'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title="Object IDs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    
    plt.show()


def plot_masked_shapes_clim(NA_shapes, CA_shapes, BS_shapes,
                                   NA_clim, CA_clim, BS_clim,
                                   list_ids, ylabel, title,vmin,vmax, savefig = False,clim=False,bars=True):
    
    '''Function to plot the masked shape values as normalized line plots in each domain together with a bar plot indicating the time 
    the object spends in the domain and optionally mean climatology'''

    cmap = cm.get_cmap("tab20b", len(list_ids))
    color_map = {idx: cmap(i) for i, idx in enumerate(list_ids)}
    
    all_shapes = [NA_shapes, CA_shapes, BS_shapes]
    clim_list = [NA_clim, CA_clim, BS_clim]
    titles = ['North Atlantic', 'Central Arctic', 'Beaufort Sea']
    
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
      
    
    for idx, (shapes,climatology, ax) in enumerate(zip(all_shapes, clim_list, axs)):
        ax2 = ax.twinx() if bars else None  #Creates second axis if condition is True
        for i, (shape, id_) in enumerate(zip(shapes, list_ids)): 
            normed_x = np.linspace(0, 1, len(shape))  #Normalizes x-coordinates
            ax.plot(normed_x, shape, color=color_map[id_], label=id_)

            if bars: 
                bar_x = (1 / 7) * i + 0.02 #Add Bars that indicate time spent in domain
                ax2.bar(bar_x, len(shape), color=color_map[id_], width=0.04)

        if clim: 
            #Add line for mean climatology 
            mean_climatology = climatology.mean(dim='time')
            ax.axhline(y=mean_climatology, color='k', linestyle='--', label=f'Mean \n Climatology')

        #Set axis limits and labels 
        ax.set_ylim(vmin,vmax)    
        ax.set_title(titles[idx])
       
        if idx == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.yaxis.set_visible(False)  
        if bars: 
            ax2.set_ylim(0, 260)
            if idx == 2:
                ax2.set_ylabel('time spent in region [h]')
            else:
                ax2.yaxis.set_visible(False)
    
    #Legend 
    axs[0].legend(bbox_to_anchor=(3.4, 1), loc='upper left', fontsize='small', title="Object IDs")

    
    for ax in axs:
        ax.set_xticks([])  
        ax.set_xlabel('')   
        
    plt.subplots_adjust(wspace=0.05)
    fig.suptitle(title, fontweight='bold',y=0.99)
    
    plt.show()


def get_shapes_and_clim(var,varname,domains,ICON_path,ICON_clim_path,objs_final,ds_moaap,ssp=False,
                        period='',startdate='',enddate='',NESM=False): 
    
    '''Function to execute masking function, returns lists of masked shapes and mean climatology for each domain
    Requires variable names and list of domains'''
    
    print(f'{ICON_path}{var}/v20250601/{var}_*.nc')
    #Load dataset and climatology
    ds = xr.open_mfdataset(f'{ICON_path}{var}/v20250601/{var}_*.nc',chunks = {'time':10})
   
    #Load climatology
    if ssp:    
        ds_clim = xr.open_dataset(f'{ICON_clim_path}/clim/{var}/{var}_clim_day_DJF_ssp_{period}.nc',chunks={'time':10})
        ds = ds.sel(time=slice(startdate, enddate))

    else: 
        ds_clim = xr.open_dataset(f'{ICON_clim_path}/clim/{var}/{var}_clim_day_DJF.nc',chunks={'time':10})

    #Get list of shape means 
    shape_list = create_masked_lists_shapes(ds[varname], domains,objs_final,ds_moaap)


    #Get list of climatology for each domain 
    NA_clim = ds_clim.where((ds_clim.lat>=67) & (ds_clim.lat<=80) & (ds_clim.lon>=-30) & (ds_clim.lon<=60))
    CA_clim = ds_clim.where((ds_clim.lat>=80) & (ds_clim.lat<=90) & (ds_clim.lon>=-180) & (ds_clim.lon<=180))
    BS_clim = ds_clim.where(
    ((ds_clim.lat >= 70) & (ds_clim.lat <= 80)) &  ((ds_clim.lon >= -180) & (ds_clim.lon <= -120)) |
        ((ds_clim.lat >= 70) & (ds_clim.lat <= 80)) & ((ds_clim.lon >= 120) & (ds_clim.lon <= 180)))

    clim_list = [NA_clim, CA_clim, BS_clim]

    
    return shape_list, clim_list 


def get_anomaly_shapes(var,varname,domains,ICON_path,ICON_clim_path,objs_final,ds_moaap,
                       ssp=False,period='',startdate='',enddate='',NESM=False): 
    '''Function to execute masking anomaly function, returns lists of masked anomalies of shapes
    Requires variable names and list of domains'''
    
    #Load dataset and climatology
    ds = xr.open_mfdataset(f'{ICON_path}{var}/v20250601/{var}_*.nc',chunks = {'time':10})
        
    #Load climatology
    if ssp:
        ds_clim = xr.open_dataset(f'{ICON_clim_path}/clim/{var}/{var}_clim_day_DJF_ssp_{period}.nc',chunks={'time': 10})
        ds = ds.sel(time=slice(startdate, enddate))
    else:
        ds_clim = xr.open_dataset(f'{ICON_clim_path}/clim/{var}/{var}_clim_day_DJF.nc',chunks={'time': 10})


    #Get list of shape anomaly means 
    anomaly_list = create_masked_lists_anomalies(ds[varname], ds_clim[varname],domains,objs_final,ds_moaap)

    return anomaly_list



def create_and_store_lists(var,var_,period,domains,ICON_path,ICON_clim_path,objs_final,ds_moaap,
                           ssp=False,startdate='',enddate='',NESM=False):
    '''Function to execute functions to create lists and store them as computed pkl files'''

    #Create shape lists and clim lists 
    shape_list, clim_list = get_shapes_and_clim(var,var_,domains,ICON_path,ICON_clim_path,
                                                objs_final,ds_moaap,ssp,period,startdate,enddate,NESM)

    #Compute so that it is not stored as dask array
    shape_list = [[da.compute() for da in sublist] for sublist in shape_list]
    clim_list = [ds.compute() for ds in clim_list]

    #Create anomaly list
    anomaly_list = get_anomaly_shapes(var,var_,domains,ICON_path,ICON_clim_path,
                                      objs_final,ds_moaap,ssp,period,startdate,enddate,NESM)
    #Compute 
    anomaly_list = [[da.compute() for da in sublist] for sublist in anomaly_list]

    #Store lists
    if ssp: 

        #Store files
        with open(f'{ICON_clim_path}shapelists/{var}_shape_list_{period}.pkl', 'wb') as f:
            pickle.dump(shape_list,f) 
        with open(f'{ICON_clim_path}shapelists/{var}_clim_list_{period}.pkl', 'wb') as f:
            pickle.dump(clim_list,f)         
   
        with open(f'{ICON_clim_path}shapelists/{var}_anomaly_list_{period}.pkl', 'wb') as f:
            pickle.dump(anomaly_list,f)

    else: 
    
        with open(f'{ICON_clim_path}shapelists/{var}_shape_list.pkl', 'wb') as f:
            pickle.dump(shape_list, f)
        with open(f'{ICON_clim_path}shapelists/{var}_clim_list.pkl', 'wb') as f:
            pickle.dump(clim_list, f)
        with open(f'{ICON_clim_path}shapelists/{var}_anomaly_list.pkl', 'wb') as f:
            pickle.dump(anomaly_list, f)
          

    return shape_list, clim_list, anomaly_list



#MOAAP INFO
def calculate_total_area(objs_plot_ids,obj_plot_times,ds_moaap,crs_rot): 
    '''Function to calculate total area covered by IVT_Object in km^2'''
    
    # Select a reference shape for mask
    ny, nx = ds_moaap['IVT_Objects'].isel(time=0).shape
    total_mask = np.zeros((ny, nx), dtype=bool)
    
    
    for tt in obj_plot_times:
        ds_moaap_sel = ds_moaap.sel(time=tt, method='nearest')
        mask_data = ds_moaap_sel.IVT_Objects.values
        
        # Find all objects to keep in this timestep
        valid_ids = [int(oid) for oid in np.unique(mask_data) if oid != 0 and str(int(oid)) in 
                     objs_plot_ids]
        if valid_ids:
            # Create a mask for any valid object in this timestep (fast)
            obj_mask = np.isin(mask_data, valid_ids)
            total_mask |= obj_mask
    
    # Load rlon/rlat
    rlon = ds_moaap['rlon'].values
    rlat = ds_moaap['rlat'].values
    rlon2d, rlat2d = np.meshgrid(rlon, rlat)
    
    
    # Transform to PlateCarree (true lat/lon)
    crs_plate = ccrs.PlateCarree()
    lonlat = crs_rot.transform_points(crs_plate, rlon2d, rlat2d)
    lon2d = lonlat[..., 0]
    lat2d = lonlat[..., 1]
    
    
    geod = Geod(ellps="WGS84")
    area_km2 = np.zeros((lat2d.shape[0]-1, lat2d.shape[1]-1))
    
    for i in range(area_km2.shape[0]):
        for j in range(area_km2.shape[1]):
            lon_corners = [lon2d[i,j], lon2d[i,j+1], lon2d[i+1,j+1], lon2d[i+1,j]]
            lat_corners = [lat2d[i,j], lat2d[i,j+1], lat2d[i+1,j+1], lat2d[i+1,j]]
            poly_area, _ = geod.polygon_area_perimeter(lon_corners, lat_corners)
            area_km2[i,j] = abs(poly_area) / 1e6  # m² → km²
    
    
    # Mask must match area grid shape (one cell smaller in each direction)
    masked = total_mask[:-1, :-1]  # crop to match area_km2 shape
    total_area_km2 = np.sum(area_km2[masked])
    
    
    return total_area_km2
        

def calculate_track_length_geopy(track_lats, track_lons):
    '''Function to calculate total track length for a given track'''
    
    total_distance = 0.0
    for i in range(len(track_lats) - 1):
        point1 = (track_lats[i], track_lons[i]) 
        point2 = (track_lats[i + 1], track_lons[i + 1])
        total_distance += distance(point1, point2).km
    return total_distance




