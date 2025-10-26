# Content: Functions for Filtering criteria for model comparison, ERA5 Lifecycle Analysis,
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
from matplotlib.ticker import MaxNLocator
import cartopy.feature
import seaborn as sns
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

##########################Load Definitions ################################
ERA5_clim_path='/work/aa0049/a271122/ERA5'

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


#Define new Domain masks
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


# Define sea ice 
ds_sic = xr.open_mfdataset("/work/bm1159/XCES/data4xces/reanalysis/reanalysis/ECMWF/IFS/ERA5/mon/seaIce/sic/r1i1p1/sic_OImon_reanalysis_era5_r1i1p1_*.nc")
ds_sic = ds_sic.sel(time=slice("1979-01-01", "2022-12-31"))
ds_sic = ds_sic.sel(time=ds_sic.sic.time.dt.month.isin([12,1,2])).mean(dim='time')


ds_sic_new = xr.open_mfdataset("/work/bm1159/XCES/data4xces/reanalysis/reanalysis/ECMWF/IFS/ERA5/mon/seaIce/sic/r1i1p1/sic_OImon_reanalysis_era5_r1i1p1_*.nc")
ds_sic_new = ds_sic_new.sel(time=slice("1979-01-01", "2022-12-31"))
ds_sic_DJF = ds_sic_new.sel(time=ds_sic_new.sic.time.dt.month.isin([12,1,2])).mean(dim='time')

#Define weights for area weighting
test_weights = xr.open_dataset('/work/aa0049/a271122/ERA5/031/new_weights2.nc')
test_weights=test_weights.assign_coords(lon=((test_weights.lon + 180) % 360) - 180).sortby(['lat','lon'])

#Select weights from nc-file 
ds_weights = xr.open_dataset('/work/aa0049/a271122/ERA5/weights.nc')
weights = ds_weights.cell_weights
weights=weights.assign_coords(lon=((weights.lon + 180) % 360) - 180).sortby(['lat','lon'])



vmin_vmax_dict = {
    'zg': (-120, 120),
    'tas': (-15, 15),
    'pr': (-0.8, 0.8),
    'hcc': (0, 100),
    'mcc': (0, 100),
    'lcc': (0, 100),
    'rlds':(-65, 65),
    'hfls':(-65, 65),
    'hfss':(-65, 65),
    'sic': (-80, 80),
    'prw': (-12,12),
    'sfcWind':(-13,13),
    'clwvi':(-0.2,0.2),
    'clivi':(-0.2,0.2)
    
}


###########################################################
#################FUNCTIONS#################################
###########################################################

def calculate_daily_clim(month_day, var_all):
    """Calculate or retrieve climatology for each month/day combination over the whole var dataset"""
    climatology_values = []

    for month, day in month_day:
        day_data = var_all.sel(time=(var_all.time.dt.month == month) & (var_all.time.dt.day == day))

        climatology_values.append(day_data.mean(dim="time"))
        
    #Create one climatology out of all month/day-combo climatologies
    climatology = xr.concat(climatology_values, dim="track_days").mean(dim="track_days")

    return climatology 


def calculate_track_length_geopy(track_lats, track_lons):
    """Calculate total track length for a given track"""
    total_distance = 0.0
    for i in range(len(track_lats) - 1):
        point1 = (track_lats[i], track_lons[i]) 
        point2 = (track_lats[i + 1], track_lons[i + 1])
        total_distance += distance(point1, point2).km
    return total_distance


def combine_times(objs_):
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
    objs_final = []
    for nr in range(len(objs_)):
        NA_time=extract_domain_times(objs_[nr],80,67,60,-30)
        BS_time=extract_domain_times(objs_[nr],80,70,120,-120,True)
        CA_time=extract_domain_times(objs_[nr],90,80,180,-180)
    
        if NA_time[0] < BS_time[0] and NA_time[0] < CA_time[0] and CA_time[0] < BS_time[0]:
            objs_final.append(objs_[nr])

    
    print(len(objs_final))
    return objs_final 

def define_df_objs(objs):
    df_objs = pd.DataFrame({
        'id': [ds.id_.item() for ds in objs],
        'color': [plt.get_cmap("tab20b")(i / len(objs))[:3] for i in range(len(objs))]
    })
    return df_objs


def create_masked_lists(objs_final,ds_moaap,var_data,north,south,east,west,BS=False):
    """Function to save masked and area weighted data points in a list, save id. Definition of domain times through center of mass """
    list_shape = []
    for i in range(1,13):
        objs_plot_ids = objs_final[i]['id_'].item()

        times = np.concatenate([extract_domain_times(objs_final[i], north, south, east, west, BS)])
        
        moaap_sel = ds_moaap.sel(time=times,method='nearest')
        data_sel = var_data.sel(time=times,method='nearest')
    
        #Create mask of chosen object and align with data grid
        mask = (moaap_sel.IVT_Objects==int(objs_plot_ids))
        aligned_mask = mask.astype(int).interp(rlat=data_sel.rlat, rlon=data_sel.rlon, method="nearest")
        aligned_mask = aligned_mask > 0
        
        #Apply the mask to the data
        filtered_data = data_sel.where(aligned_mask)
        filtered_data = filtered_data.assign_coords(lon=((filtered_data.lon + 180) % 360) - 180) #this is important to capture the entire shape 
        
        #Apply domain mask
        if BS: 
            filtered_domain = filtered_data.where(
                ((filtered_data.lat >= south) & (filtered_data.lat <= north)) &  ((filtered_data.lon >= east) & (filtered_data.lon <= 180)) |  ((filtered_data.lat >= south) & (filtered_data.lat <= north)) & ((filtered_data.lon >= -180) & (filtered_data.lon <= west)))
    
        else: 
            filtered_domain = filtered_data.where((filtered_data.lat>=south) & (filtered_data.lat<=north) & (filtered_data.lon>=west) & (filtered_data.lon<=east))

       
        filtered_weighted = filtered_domain.weighted(test_weights.cell_area)
        filtered_mean = filtered_weighted.mean(("lat","lon"),skipna=True)
 
        list_shape.append(filtered_mean)

    return list_shape


def create_masked_lists_shapes(objs_final,ds_moaap,var_data, domains):
    '''
    Function to save masked and area weighted data points, returns them in list_shape
    Definition of domain times through shapes in domain. 
    Minimum number of points to get a valid value need to be defined through min_valid_points
    '''
    list_shapes = {domain: [] for domain in domains}
    min_valid_points = 10 
    
    for i in range(1,13): #(1,11) for 1984-2014
        times_i = objs_final[i].times
        #Load MOAAP shapes for all event timesteps
        moaap_sel = ds_moaap.sel(time=times_i, method='nearest')
        mask = (moaap_sel.IVT_Objects == int(objs_final[i].id_.item()))
    
        #Load variable data for the same timesteps
        data_sel = var_data.sel(time=times_i, method='nearest')
    
        # Align mask with data grid
        aligned_mask = mask.astype(int).interp(rlat=data_sel.rlat, rlon=data_sel.rlon, method="nearest")
        aligned_mask = aligned_mask > 0
    
        # Apply the mask on the variable data
        filtered_data = data_sel.where(aligned_mask)
        filtered_data = filtered_data.assign_coords(lon=((filtered_data.lon + 180) % 360) - 180)

        for domain in domains:
            #Select domain
            if domain == "BeaufortSiberianSea":
                # Apply domain mask
                filtered_domain = filtered_data.where(
                            ((filtered_data.lat >= mask_domains[domain]["south"]) & (filtered_data.lat <= mask_domains[domain]["north"])) &  ((filtered_data.lon >= mask_domains[domain]["east"]) & (filtered_data.lon <= 180)) |  ((filtered_data.lat >= mask_domains[domain]["south"]) & (filtered_data.lat <= mask_domains[domain]["north"])) & ((filtered_data.lon >= -180) & (filtered_data.lon <= mask_domains[domain]["west"])))       
            else: 
                filtered_domain = filtered_data.where((filtered_data.lat>=mask_domains[domain]["south"]) & (filtered_data.lat<=mask_domains[domain]["north"]) & (filtered_data.lon>=mask_domains[domain]["west"]) & (filtered_data.lon<=mask_domains[domain]["east"]))
        
            #Check if filtered_domain has at least min_valid_points
            num_valid_points = filtered_domain.count()
        
            if num_valid_points >= min_valid_points:
        
                #Create an area weighted mean using test_weights and add mean value to the list
                filtered_weighted = filtered_domain.weighted(test_weights.cell_area)
                filtered_mean = filtered_weighted.mean(("lat","lon"),skipna=True)

                #Remove NaN values
                valid_filtered_mean = filtered_mean.dropna(dim='times', how='all')
         
                list_shapes[domain].append(valid_filtered_mean)

    # Return lists for all domains
    return [list_shapes[domain] for domain in domains]


def create_masked_lists_anomalies(objs_final,ds_moaap,var_data,clim,north,south,east,west,BS=False): 
    ###Anomalies
    """Function to save masked and area weighted data points in a list, save id"""
    list_shape = []
    for i in range(1,13): #(1,11) for 1984-2014
        objs_plot_ids = objs_final[i]['id_'].item()

        times = np.concatenate([extract_domain_times(objs_final[i], north, south, east, west, BS)])
        
        #Select climatology for specific calendar days
        month_day = list(set((time.month, time.day) for time in pd.to_datetime(times)))
        clim_daysofyear = clim.where(clim['time'].dt.strftime('%m-%d').isin([f"{month:02d}-{day:02d}" for month,day in month_day]), drop=True)
                
        moaap_sel = ds_moaap.sel(time=times,method='nearest')
        data_sel = var_data.sel(time=times,method='nearest') - clim_daysofyear.mean(dim='time')
    
        #Create mask of chosen object and align with data grid
        mask = (moaap_sel.IVT_Objects==int(objs_plot_ids))
        aligned_mask = mask.astype(int).interp(rlat=data_sel.rlat, rlon=data_sel.rlon, method="nearest")
        aligned_mask = aligned_mask > 0
        
        #Apply the mask to the data
        filtered_data = data_sel.where(aligned_mask)
        filtered_data = filtered_data.assign_coords(lon=((filtered_data.lon + 180) % 360) - 180) #this is important to capture the entire shape         
        
        #Apply domain mask
        if BS: 
            filtered_domain = filtered_data.where(
                ((filtered_data.lat >= south) & (filtered_data.lat <= north)) &  ((filtered_data.lon >= east) & (filtered_data.lon <= 180)) |  ((filtered_data.lat >= south) & (filtered_data.lat <= north)) & ((filtered_data.lon >= -180) & (filtered_data.lon <= west)))
    
        else: 
            filtered_domain = filtered_data.where((filtered_data.lat>=south) & (filtered_data.lat<=north) & (filtered_data.lon>=west) & (filtered_data.lon<=east))

        
        filtered_weighted = filtered_domain.weighted(test_weights.cell_area)
        filtered_mean = filtered_weighted.mean(("lat","lon"),skipna=True)
        
        list_shape.append(filtered_mean) 

    return list_shape
    
def create_masked_lists_anomalies_NEW(objs_final,ds_moaap,var_data,clim,domains):
    ###Anomalies
    """Function to save masked and area weighted data points in a list, save id"""
    list_shapes = {domain: [] for domain in domains}
    min_valid_points = 10 

    for i in range(1,13): #(1,11) for 1984-2014
        objs_plot_ids = objs_final[i]['id_'].item()

        time_i=objs_final[i].times
        
        #Select climatology for specific calendar days
        month_day = list(set((time.month, time.day) for time in pd.to_datetime(time_i)))
        clim_daysofyear = clim.where(clim['time'].dt.strftime('%m-%d').isin([f"{month:02d}-{day:02d}" for month,day in month_day]), drop=True)
                
        moaap_sel = ds_moaap.sel(time=time_i,method='nearest')
        data_sel = var_data.sel(time=time_i,method='nearest') - clim_daysofyear.mean(dim='time')
    
        #Create mask of chosen object and align with data grid
        mask = (moaap_sel.IVT_Objects==int(objs_plot_ids))
        aligned_mask = mask.astype(int).interp(rlat=data_sel.rlat, rlon=data_sel.rlon, method="nearest")
        aligned_mask = aligned_mask > 0
        
        #Apply the mask to the data
        filtered_data = data_sel.where(aligned_mask)
        filtered_data = filtered_data.assign_coords(lon=((filtered_data.lon + 180) % 360) - 180)
  

        for domain in domains:
            #Apply domain mask
            if domain == "BeaufortSiberianSea":
                    # Apply domain mask
                    filtered_domain = filtered_data.where(
                                ((filtered_data.lat >= mask_domains[domain]["south"]) & (filtered_data.lat <= mask_domains[domain]["north"])) &  ((filtered_data.lon >= mask_domains[domain]["east"]) & (filtered_data.lon <= 180)) |  ((filtered_data.lat >= mask_domains[domain]["south"]) & (filtered_data.lat <= mask_domains[domain]["north"])) & ((filtered_data.lon >= -180) & (filtered_data.lon <= mask_domains[domain]["west"])))       
            else: 
                    filtered_domain = filtered_data.where((filtered_data.lat>=mask_domains[domain]["south"]) & (filtered_data.lat<=mask_domains[domain]["north"]) & (filtered_data.lon>=mask_domains[domain]["west"]) & (filtered_data.lon<=mask_domains[domain]["east"]))
            
            #Check if filtered_domain has at least min_valid_points
            num_valid_points = filtered_domain.count()
        
            if num_valid_points >= min_valid_points:
        
                #Create an area weighted mean using test_weights and add mean value to the list
                filtered_weighted = filtered_domain.weighted(test_weights.cell_area)
                filtered_mean = filtered_weighted.mean(("lat","lon"),skipna=True)

                #Remove NaN values
                valid_filtered_mean = filtered_mean.dropna(dim='times', how='all')
         
         
                list_shapes[domain].append(valid_filtered_mean)

    # Return lists for all domains
    return [list_shapes[domain] for domain in domains]


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

def extract_dates_freq(objs):
    dates = []
    length = []
    for i in range(len(objs)):
        dates.append(np.datetime64(objs[i]['times'][0].values))
        length.append(len(objs[i]['times']))
    return dates, length


def filter_by_size(objs, threshold=0.03e6):
    filtered_objs = []
    for obj in objs:
        size_mean = obj.size.mean(dim='times').item()
        if size_mean > threshold:
            filtered_objs.append(obj)
    return filtered_objs

def get_month_day(objs_time):
    #Extract month, day pairs from track times
    
    track_times_pd = pd.to_datetime(objs_time.values)
    month_day = list(set((time.month, time.day) for time in track_times_pd))
    return month_day

def get_calendar_days(NA_times,CA_times,BS_times):
    """Function that returns combined calender dates of combined intrusions in different sectors"""
    
    NA_month_day = list(set((time.month, time.day) for time in pd.to_datetime(NA_times)))
    CA_month_day = list(set((time.month, time.day) for time in pd.to_datetime(CA_times)))
    BS_month_day = list(set((time.month, time.day) for time in pd.to_datetime(BS_times)))

    return NA_month_day, CA_month_day, BS_month_day 
def get_anomaly_shapes(data_var,var,varname,domains): 
    '''Function to execute masking anomaly function, returns lists of masked anomalies of shapes
    Requires variable names and list of domains'''

    #Load climatology
    ds_clim = xr.open_dataset(f'{ERA5_clim_path}/{var}/clim/{var}_clim_day_DJF.nc',chunks={'time': 10})

    #Get list of shape anomaly means 
    anomaly_list = create_masked_lists_anomalies_NEW(data_var, ds_clim[varname],domains)

    #Save list as pkl file once 
    with open(f'{ERA5_clim_path}/shapelists/{var}_anomaly_list.pkl', 'wb') as f:
        pickle.dump(anomaly_list, f)
         
    return anomaly_list
    
def get_shapes_and_clim(data,var,varname,domains): 
    '''Function to execute masking function, returns lists of masked shapes and mean climatology for each domain
    Requires variable names and list of domains'''

    #Get list of shape means 
    shape_list = create_masked_lists_shapes(data, domains)

    #Load climatology 
    clim_in = xr.open_dataset(f'{ERA5_clim_path}/{var}/clim/{var}_clim_day_DJF.nc')
    ds_clim = clim_in.assign_coords(lon=((clim_in.lon + 180) % 360) - 180).sortby(['lat','lon'])
    
    #Get list of climatology for each domain (North Atlantic, Central Arctic, Beaufort/Siberian Sea)  
    NA_clim = ds_clim.where((ds_clim.lat>=67) & (ds_clim.lat<=80) & (ds_clim.lon>=-30) & (ds_clim.lon<=60))
    CA_clim = ds_clim.where((ds_clim.lat>=80) & (ds_clim.lat<=90) & (ds_clim.lon>=-180) & (ds_clim.lon<=180))
    BS_clim = ds_clim.where(
    ((ds_clim.lat >= 70) & (ds_clim.lat <= 80)) &  ((ds_clim.lon >= -180) & (ds_clim.lon <= -120)) |  ((ds_clim.lat >= 70) & (ds_clim.lat <= 80)) & ((ds_clim.lon >= 120) & (ds_clim.lon <= 180)))

    clim_list = [NA_clim, CA_clim, BS_clim]

    #Save lists as pkl files once 
    with open(f'{ERA5_clim_path}/shapelists/{var}_shape_list.pkl', 'wb') as f:
        pickle.dump(shape_list, f)
    with open(f'{ERA5_clim_path}/shapelists/{var}_clim_list.pkl', 'wb') as f:
        pickle.dump(clim_list, f)
     
    return shape_list, clim_list 
    
def plot_tracks_2(objs_,df_objs_):

    fig,ax=plt.subplots(1,1,subplot_kw={'projection': ccrs.Orthographic(0,90)},figsize=(5,5))
    objs_nr=len(objs_)
    
    for ii in range(len(objs_)):
        track=objs_[ii].track.values
        
        track_lats = np.array([x.lat for x in track])
        track_lons = np.array([x.lon for x in track])    
        
    
        obj_id=objs_[ii]['id_'].item()
        obj_color=df_objs_[  df_objs_['id'] == obj_id].color.item()
        ax.text(track_lons[-1],track_lats[-1],obj_id,fontsize=8,\
                transform=crs_rot,color=obj_color,zorder=11)
        ax.text(track_lons[0],track_lats[0],str(objs_[ii]['times'][0].values.astype('datetime64[h]')),fontsize=8,\
                transform=crs_rot,color=obj_color,zorder=11)
        ax.scatter(track_lons[0],track_lats[0],facecolor=obj_color,edgecolor='k',transform=crs_rot)
        ax.plot(track_lons,track_lats,color=obj_color,transform=crs_rot)  
    
    duration_mean=np.array([len(obj.times) for obj in objs_]).mean()
    
    #this marks the domain (North Atlantic, Central Arctic and Beaufort area) 
    for domain_name, domain_data in mask_domains.items():
        domain_data["mask"].plot.contour(levels=1,colors=domain_data["color"],linewidths=1.5,linestyles='--',zorder=4,ax=ax,transform=ccrs.PlateCarree())
    
    ax.text(0.02,0.02,'nr: {}, duration: {:d}h'.format(objs_nr,int(np.round(duration_mean))),transform=ax.transAxes,\
               bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    ax.coastlines()
    ax.set_extent([-180,180, 58, 90], crs=ccrs.PlateCarree())
    ax.gridlines(linewidth=0.5,color='dimgray',alpha=0.4,zorder=2,draw_labels=True)
    ax.set_title('ERA5 1998-2022',loc='left',y=1)
    fig.suptitle('IVT object tracks crossing the three marked sectors in DJF (dots = start)', fontweight='bold',y=0.95)

def plot_tracks_PC(objs_, df_objs_):
    """Plots tracks in PlateCarree projection with dateline-aware splitting."""

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.Orthographic(0, 90)}, figsize=(7,7))
    
    for ii in range(len(objs_)):
        latitudes = objs_[ii].geo_lat.values  
        longitudes = objs_[ii].geo_lon.values

        #Split track at the dateline
        track_lons, track_lats = split_tracks_180(longitudes, latitudes)

        #Plot full track
        obj_id = objs_[ii]['id_'].item()
        obj_color = df_objs_[df_objs_['id'] == obj_id].color.item()
        ax.plot(track_lons, track_lats, color=obj_color, transform=ccrs.PlateCarree())  
        ax.scatter(track_lons[0], track_lats[0], facecolor=obj_color, edgecolor='k', transform=ccrs.PlateCarree())
        ax.text(track_lons[0], track_lats[0], str(objs_[ii]['times'][0].values.astype('datetime64[h]')), fontsize=8,
                transform=ccrs.PlateCarree(), color=obj_color, zorder=11)
        ax.text(track_lons[-1], track_lats[-1], obj_id, fontsize=8,transform=ccrs.PlateCarree(), color=obj_color, zorder=11)
    
    for domain_name, domain_data in mask_domains.items():
        domain_data["mask"].plot.contour(levels=1,colors=domain_data["color"],linewidths=1.5,linestyles='--',zorder=4,ax=ax,transform=ccrs.PlateCarree())
    
    ax.coastlines()
    ax.set_extent([-180, 180, 58, 90], crs=ccrs.PlateCarree())
    ax.gridlines(linewidth=0.5, color='dimgray', alpha=0.4, zorder=2, draw_labels=True)
    ax.set_title('ERA5 1998-2022', loc='left',y=1)
    fig.suptitle('IVT object tracks crossing the three marked sectors in DJF (dots = start)', fontweight='bold', y=0.96)
    
def plot_all_domain_means(objs_final,var_NA,var_CA,var_BS,con_NA,con_CA,con_BS,title,label_cbar,vmin=None,vmax=None,tracks_on=True,anomaly=True,contour_on=True,sic=False):

    #Select colormap, diverging if anomaly and sequential for total values
    cmap = 'RdBu_r' if anomaly else 'RdYlBu_r'
    fig, axs = plt.subplots(1,3,subplot_kw={'projection': ccrs.Orthographic(0,90)},figsize=(16,4))

    #Get track data
    #Compute tracks for adding them to plots later 
    precomputed_tracks = [split_tracks_180(obj.geo_lon.values, obj.geo_lat.values) for obj in objs_final]
    tracks = precomputed_tracks
    
    #Plot mean for when tracks are within selected regions 
    plot_domain_mean(var_NA,con_NA,axs[0],"NorthAtlantic",title+" North Atlantic",label_cbar,objs_final,cmap,tracks,vmin,vmax,tracks_on,contour_on,sic)
    plot_domain_mean(var_CA,con_CA,axs[1],"CentralArctic",title+" Central Arctic",label_cbar,objs_final,cmap,tracks,vmin,vmax,tracks_on,contour_on,sic)
    plot_domain_mean(var_BS,con_BS,axs[2],"BeaufortSiberianSea",title+" Beaufort Sea",label_cbar,objs_final,cmap,tracks,vmin,vmax,tracks_on,contour_on,sic)
    
    for aa in axs:
        
        aa.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
        aa.gridlines(linestyle='--',zorder=3)
        aa.coastlines(zorder=2)
        aa.add_feature(cartopy.feature.LAND, color='lightgray',zorder=0, edgecolor='None') 
        
        #create circular image
        center, radius = [0.5, 0.5], 0.55
        circle = mpath.Path(np.vstack([np.sin(np.linspace(0, 2*np.pi, 100)), np.cos(np.linspace(0, 2*np.pi, 100))]).T * radius + center)
        aa.set_boundary(circle, transform = aa.transAxes)


def plot_anom_and_contour(var_sel,var_clim,contour_sel,title):
    # Plot var_mean anomaly for selected events 
    fig, ax = plt.subplots(1, 1,subplot_kw={'projection': ccrs.Orthographic(0,90)})
    
    (var_sel.mean(dim='times')-var_clim).plot(ax=ax,transform=ccrs.PlateCarree(),cmap='RdBu_r')
    
    contour = contour_sel.plot.contour(levels=12,ax=ax,transform=ccrs.PlateCarree(),colors='k',linewidths=0.8)
    ax.clabel(contour, fmt='%1.0f', inline=True, fontsize=10)
    #cbar = plt.colorbar(contour,ax=ax, orientation = 'vertical', pad=0.05)
    #cbar.set_label('Geopotential height (m)')
     
    ds_sic.sic.plot.contour(levels=1, ax = ax, transform = ccrs.PlateCarree(), colors = 'indigo', linewidths = 1.2, linestyles = '-')
    
    ax.set_extent([-180, 180, 55, 90], crs=ccrs.PlateCarree())
    ax.gridlines(linestyle='--',zorder=3)
    ax.coastlines(zorder=2)
    ax.add_feature(cartopy.feature.LAND, color='lightgray',zorder=0, edgecolor='None') 
    
    
    center, radius = [0.5, 0.5], 0.5
    circle = mpath.Path(np.vstack([np.sin(np.linspace(0, 2*np.pi, 100)), np.cos(np.linspace(0, 2*np.pi, 100))]).T * radius + center)
    ax.set_boundary(circle, transform = ax.transAxes)
    ax.set_title(f"{title} anomaly for WAIS in 1979-2022",y=1.05)

def plot_domain_mean(ds_domain_mean,ds_contour_mean,axs,domain,title,
                     label_cbar,objs_,cmap,tracks,vmin=None,vmax=None,
                     tracks_on=True,contour_on=True,sic=False):
    #plot variable
    variable =  ds_domain_mean.plot(ax=axs,transform=ccrs.PlateCarree(),cmap=cmap,vmin=vmin,vmax=vmax, add_colorbar = False)
   
    # Add custom colorbar
    colorbar = plt.colorbar(variable, ax=axs, orientation='vertical', pad=0.06, shrink=0.8)
    colorbar.set_label(label_cbar, fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    
    #Add additional data
    # add contour
    if contour_on==True: 
        contour = ds_contour_mean.plot.contour(levels=11,ax=axs,transform=ccrs.PlateCarree(),colors='k',linewidths=0.5)
        axs.clabel(contour, fmt='%1.0f', inline=True, fontsize=7)
        
    #add domain mask
    mask_domains[domain]["mask"].plot.contour(levels=1,colors=mask_domains[domain]["color"],\
        linewidths=1.5,linestyles='--',zorder=4,ax=axs,transform=ccrs.PlateCarree())

    #add tracks
    if tracks_on==True: 
        for lon, lat in tracks:
            axs.plot(lon, lat, color='gray', alpha=0.5, transform=ccrs.PlateCarree())
            axs.scatter(lon[0], lat[0], facecolor='gray', edgecolor='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=1)

    #add sea ice edge
    if sic == True: 
        ds_sic.sic.plot.contour(levels=1, ax = axs, transform = ccrs.PlateCarree(), colors = 'indigo', linewidths = 1, linestyles = '-')

    axs.set_title(f"{title} sector",y=1.05)

def plot_domain_mean_new(ds_domain_mean,ds_contour_mean,
                         axs,domain,title,label_cbar,objs_,
                         cmap,tracks,vmin=None,vmax=None,
                         tracks_on=True,contour_on=True,sic=False):
    #plot variable
    variable =  ds_domain_mean.plot(ax=axs,transform=ccrs.PlateCarree(),cmap=cmap,vmin=vmin,vmax=vmax, add_colorbar = False)
   
    # Add custom colorbar
    colorbar = plt.colorbar(variable, ax=axs, orientation='horizontal', pad=0.11, shrink=0.8, extend = 'both')
    colorbar.set_label(label_cbar, fontsize=18)
    colorbar.ax.tick_params(labelsize=15)
    
    #Add additional data
    # add contour
    if contour_on==True: 
        contour = ds_contour_mean.plot.contour(levels=7,ax=axs,transform=ccrs.PlateCarree(),colors='k',linewidths=0.5)
        axs.clabel(contour, fmt='%1.0f', inline=True, fontsize=7)
        
    #add domain mask
    mask_domains[domain]["mask"].plot.contour(levels=1,colors=mask_domains[domain]["color"],\
        linewidths=2.8,linestyles='--',zorder=6,ax=axs,transform=ccrs.PlateCarree())

    mask_domains["NorthAtlantic"]["mask"].plot.contour(levels=1,colors=mask_domains["NorthAtlantic"]["color"],\
        linewidths=1.8,linestyles='--',zorder=4,ax=axs,transform=ccrs.PlateCarree())

    mask_domains["BeaufortSiberianSea"]["mask"].plot.contour(levels=1,colors=mask_domains["BeaufortSiberianSea"]["color"],\
        linewidths=1.8,linestyles='--',zorder=4,ax=axs,transform=ccrs.PlateCarree())

    #add tracks
    if tracks_on==True: 
        for lon, lat in tracks:
            axs.plot(lon, lat, color='gray', alpha=0.7, transform=ccrs.PlateCarree())
            axs.scatter(lon[0], lat[0], facecolor='gray', edgecolor='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=1)

    #add sea ice edge
    if sic == True: 
        ds_sic.sic[0,:,:].plot.contour(levels=1, ax = axs, transform = ccrs.PlateCarree(), colors = 'cyan', linewidths = 1, linestyles = '-')

    axs.set_title(f"{title} sector",y=1.1)

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
                bar_x = (1 / 12) * i + 0.02 #Add Bars that indicate time spent in domain
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


def plot_var_and_contour(var_sel,contour_sel,title):
    # Plot var_mean for selected events 
    fig, ax = plt.subplots(1, 1,subplot_kw={'projection': ccrs.Orthographic(0,90)})
    
    var_sel.mean(dim='times').plot(ax=ax,transform=ccrs.PlateCarree(),cmap='RdYlBu_r')
    
    contour = contour_sel.mean(dim='times').plot.contour(levels=12,ax=ax,transform=ccrs.PlateCarree(),colors='k',linewidths=0.8)
    ax.clabel(contour, fmt='%1.0f', inline=True, fontsize=10)
    ds_sic.sic.plot.contour(levels=1, ax = ax, transform = ccrs.PlateCarree(), colors = 'indigo', linewidths = 1.2, linestyles = '-')
    #cbar = plt.colorbar(contour,ax=ax, orientation = 'vertical', pad=0.05)
    #cbar.set_label('Geopotential height (m)')
    
    ax.set_extent([-180, 180, 55, 90], crs=ccrs.PlateCarree())
    ax.gridlines(linestyle='--',zorder=3)
    ax.coastlines(zorder=2)
    ax.add_feature(cartopy.feature.LAND, color='lightgray',zorder=0, edgecolor='None') 
    


    center, radius = [0.5, 0.5], 0.5
    circle = mpath.Path(np.vstack([np.sin(np.linspace(0, 2*np.pi, 100)), np.cos(np.linspace(0, 2*np.pi, 100))]).T * radius + center)
    ax.set_boundary(circle, transform = ax.transAxes)
    ax.set_title(f"Mean {title} for WAIS in 1979-2022",y=1.05)



def plot_scatter_domains(NA_shapes, CA_shapes, BS_shapes, list_ids, ylabel, title, savefig = False):
    """ Function to create a scatter plot for each shape mean value per timestep for all objects together with the median value of all shapes of one object"""
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

def plot_box_years(anomalies_84_14,anomalies_15_39,anomalies_40_69,maintitle,ylabel,ymin,ymax): 
    #plt.rcParams.update({
    #'axes.titlesize': 24,        # Title size
    #'axes.labelsize': 20,         # Axis label size
    #'lines.linewidth': 3,         # Line width
    #'lines.markersize': 10,       # Marker size for lines
    #'xtick.labelsize': 15,        # X-tick label size
    #'ytick.labelsize': 15         # Y-tick label size
    #})
    titles = ['North Atlantic', 'Central Arctic', 'Beaufort Sea']

    fig, axs = plt.subplots(1, 1, figsize=(8, 10), sharey=True)
    
    # Define colors for each dataset
    #colors = ['#377EB8', '#FF7F00', '#4DAF4A', '#984EA3']
    #colors = ["#1B9E77","#D95F02", "#7570B3"]
    colors = ["darkblue","darkred", "#D95F02"]
    #colors = ["#6a8daf", "#f28c8c", "#fff4a3"]

    
    bp = axs.boxplot(
        [anomalies_84_14,anomalies_15_39,anomalies_40_69],
        positions=[1, 2, 3], 
        patch_artist=True,  # Allows setting box colors
        showmeans=False
    )

    # Set colors for boxes, whiskers, caps, medians, and outliers
    for i, (box, color) in enumerate(zip(bp['boxes'], colors)):
        box.set(facecolor=color, alpha=0.6, edgecolor='black')

        # Whiskers and caps
        for whisker in bp['whiskers'][2 * i:2 * i + 2]:
            whisker.set(color=color, linewidth=1.2)
        for cap in bp['caps'][2 * i:2 * i + 2]:
            cap.set(color=color, linewidth=1.2)

        # Medians
        bp['medians'][i].set(color='black', linewidth=1.5)

        # Outlier points
        bp['fliers'][i].set(marker='o', markerfacecolor=color, markeredgecolor=color, alpha=0.6, markersize=5)

    data = [anomalies_84_14, anomalies_15_39, anomalies_40_69]
    labels = ['NA', 'CA', 'BS']
    
    for label, group in zip(labels, data):
        group_clean = [x for x in group if x is not None and not np.isnan(x)]  # remove NaNs if needed
        mean_val = np.mean(group_clean)
        median_val = np.median(group_clean)
        min_val = np.min(group_clean)
        max_val = np.max(group_clean)
        std_val = np.std(group_clean)

        print(f"{label} — Mean: {mean_val:.4f}, Median: {median_val:.4f}, "
          f"Min: {min_val:.4f}, Max: {max_val:.4f}, Std Dev: {std_val:.4f}")
        
    # Set axis labels and formatting
    axs.set_xticks([1, 2, 3])
    axs.set_xticklabels(['North Atlantic', 'Central Arctic', 'Beaufort Sea'])#, rotation=90)
    axs.set_ylim([ymin, ymax])

    axs.set_ylabel(ylabel)
    
    # Create a legend manually
    #legend_labels = ["1985-2014", "2015-2039", "2040-2069", "2070-2099"]
    #legend_patches = [plt.Line2D([0], [0], color=color, marker='s', markersize=10, linestyle='None') for color in colors]
    
    
    plt.subplots_adjust(wspace=0.05)
    fig.suptitle(maintitle, fontweight='bold',y=0.94,fontsize=20)

    plt.savefig(f"boxplot_{maintitle}.png",dpi=300)
    plt.show()


def plot_box_years_new(anomalies_NA,anomalies_CA,anomalies_BS,clim_list,var,maintitle,ylabel,ymin,ymax,clim_min,clim_max): 
    #plt.rcParams.update({
    #'axes.titlesize': 24,        # Title size
    #'axes.labelsize': 21,         # Axis label size
    #'lines.linewidth': 3,         # Line width
    #'lines.markersize': 10,       # Marker size for lines
    #'xtick.labelsize': 19,        # X-tick label size
    #'ytick.labelsize': 16         # Y-tick label size
    #})
    titles = ['North Atlantic', 'Central Arctic', 'Beaufort Sea']

    fig, axs = plt.subplots(1, 1, figsize=(8, 9), sharey=True)
    
    # Define colors for each dataset
    #colors = ['#377EB8', '#FF7F00', '#4DAF4A', '#984EA3']
    #colors = ["#1B9E77","#D95F02", "#7570B3"]
    colors = ["darkblue","darkred", "#D95F02"]
    #colors = ["#6a8daf", "#f28c8c", "#fff4a3"]
    ax2 = axs.twinx()
    positions=[1, 2, 3]

    if var=='pr':
        clim_NA = clim_list[0][var].mean(dim='time').mean(dim=('lat','lon')).values*3600
        clim_CA = clim_list[1][var].mean(dim='time').mean(dim=('lat','lon')).values*3600
        clim_BS = clim_list[2][var].mean(dim='time').mean(dim=('lat','lon')).values*3600
        
        print(f' Clim NA: {clim_NA:.2f} \n CA: {clim_CA:.2f} \n BS: {clim_BS:.2f}')
        ax2.hlines([clim_NA,
                    clim_CA,
                    clim_BS],
                   [p - 0.3 for p in positions],  # xmin per box
                   [p + 0.3 for p in positions],  # xmax per box
                    colors='black', linewidth=2,linestyle='--')
    else:
        clim_NA = clim_list[0][var].mean(dim='time').mean(dim=('lat','lon')).values
        clim_CA = clim_list[1][var].mean(dim='time').mean(dim=('lat','lon')).values
        clim_BS = clim_list[2][var].mean(dim='time').mean(dim=('lat','lon')).values
        
        
        ax2.hlines([clim_NA,
                    clim_CA, 
                    clim_BS],
                       [p - 0.3 for p in positions],  # xmin per box
                       [p + 0.3 for p in positions],  # xmax per box
                        colors='black', linewidth=2,linestyle='--')



        print(f' Clim NA: {clim_NA:.2f} \n CA: {clim_CA:.2f} \n BS: {clim_BS:.2f}')
    bp = axs.boxplot(
        [anomalies_NA,anomalies_CA,anomalies_BS],
        positions=[1, 2, 3], 
        patch_artist=True,  # Allows setting box colors
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=7),
        whis = (1, 99),
    )

    # Set colors for boxes, whiskers, caps, medians, and outliers
    for i, (box, color) in enumerate(zip(bp['boxes'], colors)):
        box.set(facecolor=color, alpha=0.9, edgecolor='black')

        # Whiskers and caps
        for whisker in bp['whiskers'][2 * i:2 * i + 2]:
            whisker.set(color=color, linewidth=1.2)
        for cap in bp['caps'][2 * i:2 * i + 2]:
            cap.set(color=color, linewidth=1.2)

        # Medians
        bp['medians'][i].set(color='black', linewidth=1.0)

        # Outlier points
        bp['fliers'][i].set(marker='o', markerfacecolor=color, markeredgecolor=color, alpha=0.9, markersize=5)

    data = [anomalies_NA, anomalies_CA, anomalies_BS]
    labels = ['NA', 'CA', 'BS']
    
    for label, group in zip(labels, data):
        group_clean = [x for x in group if x is not None and not np.isnan(x)]  # remove NaNs if needed
        mean_val = np.mean(group_clean)
        median_val = np.median(group_clean)
        min_val = np.min(group_clean)
        max_val = np.max(group_clean)
        std_val = np.std(group_clean)

        print(f"{label} — Mean: {mean_val:.4f}, Median: {median_val:.4f}, "
          f"Min: {min_val:.4f}, Max: {max_val:.4f}, Std Dev: {std_val:.4f}")
        
    # Set axis labels and formatting
    axs.set_xticks([1, 2, 3])
    axs.set_xticklabels(['North Atlantic', 'Central Arctic', 'Beaufort Sea'])#, rotation=90)
    axs.set_ylim([ymin, ymax])

    axs.set_ylabel(ylabel)
    
    ax2.set_ylim([clim_min,clim_max])
    ax2.set_ylabel("Climatological Mean")


    plt.subplots_adjust(wspace=0.05)
    fig.suptitle(maintitle, fontweight='bold',y=0.99,fontsize=20)

    plt.tight_layout()
    
    plt.savefig(f"{ERA5_clim_path}/boxplot_{maintitle}.png",dpi=300,bbox_inches='tight')
    plt.show()

def plot_box_and_clim(anomalies_84_14,anomalies_15_39,anomalies_40_69,anomalies_70_99,
                      clim_list_84_14,clim_list_15_39,clim_list_40_69,clim_list_70_99,
                      var,maintitle,ylabel,ymin,ymax,ICON_clim_path,clim_min,clim_max): 
    '''Function to plot development over the centuries in one model as box plot with adjusted colors, mean and median and outliers 
    defined as outside of 98% of the points as well as climatology in a second axis'''


    #plt.rcParams.update({
    #'axes.titlesize': 24,        # Title size
    #'axes.labelsize': 18,         # Axis label size
    #'lines.linewidth': 3,         # Line width
    #'lines.markersize': 10,       # Marker size for lines
    #'xtick.labelsize': 15,        # X-tick label size
    #'ytick.labelsize': 14         # Y-tick label size
    #})
    
    
    titles = ['North Atlantic', 'Central Arctic', 'Beaufort Sea']

    fig, axs = plt.subplots(1, 3, figsize=(8, 10), sharey=True)


    #Here, we need to define a second axis that is only visible in the third box on the right side 
    #second axis shows climatology as a line 
    
    # Define colors for each dataset
    colors = ["darkblue","darkred", "#D95F02"]
    alpha_ = [1,0.8,0.6,0.4]
    
    for idx, (title, ax) in enumerate(zip(titles, axs)):
        ax2 = ax.twinx()
        positions=[1, 2, 3, 4]
        if var=='pr':
            clim_84 = clim_list_84_14[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values*3600
            clim_15 = clim_list_15_39[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values*3600
            clim_40 = clim_list_40_69[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values*3600
            clim_70 = clim_list_70_99[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values*3600
            
            ax2.hlines([clim_84,
                        clim_15,
                        clim_40,
                        clim_70],
                       [p - 0.3 for p in positions],  # xmin per box
                       [p + 0.3 for p in positions],  # xmax per box
                        colors='black', linewidth=2,linestyle='--')
        else:    
            ax2.hlines([clim_list_84_14[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values,
                        clim_list_15_39[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values,
                        clim_list_40_69[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values,
                        clim_list_70_99[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values],
                       [p - 0.3 for p in positions],  # xmin per box
                       [p + 0.3 for p in positions],  # xmax per box
                        colors='black', linewidth=2,linestyle='--')

        
        # Create boxplots
        bp = ax.boxplot(
            [anomalies_84_14[idx],anomalies_15_39[idx],anomalies_40_69[idx],anomalies_70_99[idx]],
            positions=[1, 2, 3, 4], 
            patch_artist=True,  # Allows setting box colors
            showmeans=True,
            meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=7),
            whis = (1, 99)
        )
        color = colors[idx]
    
        
    
        # Set colors for boxes, whiskers, caps, medians, and outliers
        for i, (box, alp) in enumerate(zip(bp['boxes'], alpha_)):
            box.set(facecolor=color, alpha=alp, edgecolor='black')
    
            # Whiskers and caps
            for whisker in bp['whiskers'][2 * i:2 * i + 2]:
                whisker.set(color=color, linewidth=1.2)
            for cap in bp['caps'][2 * i:2 * i + 2]:
                cap.set(color=color, linewidth=1.2)
    
            # Medians
            bp['medians'][i].set(color='black', linewidth=1.0)
            bp['medians'][i].set_clip_on(True)


            # Skip fliers since we're using whis='range'
            # Outlier points
            bp['fliers'][i].set(marker='o', markerfacecolor=color, markeredgecolor=color, alpha=alp, markersize=5)
    

    
        # Set axis labels and formatting
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["", "", "", ""], rotation=90)

        ax.set_xticklabels(["1985-2014", "2015-2039", "2040-2069", "2070-2099"], rotation=90)
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel(title, labelpad=20, y=-0.25)

        #Set second axis values
        ax2.set_ylim([clim_min,clim_max])
    
    
        if idx == 0:
            ax.set_ylabel(ylabel)
            ax2.yaxis.set_visible(False)  
    
        elif idx == 2: 
            ax2.set_ylabel("")#Climatological Mean")  
        else:
            ax.yaxis.set_visible(False) 
            ax2.yaxis.set_visible(False)   
        
    plt.subplots_adjust(wspace=0.05)
    fig.suptitle(maintitle, fontweight='bold',y=1.01,fontsize=20)


    safe_title = re.sub(r'[^\w\-.]', '_', maintitle)  # Replace unsafe characters
    plt.savefig(f'{ICON_clim_path}/{safe_title}_clim.png',dpi=300,bbox_inches='tight')
    
    plt.show()



def plot_violin_years(anomalies_84_14, anomalies_15_39, anomalies_40_69, maintitle, ylabel, ymin, ymax):
    """Plot violin plots"""
    data = [anomalies_84_14, anomalies_15_39, anomalies_40_69]
    labels = ['North Atlantic', 'Central Arctic', 'Beaufort Sea']
    colors = ["darkblue","darkred", "#D95F02"] # pastel versions

    plt.figure(figsize=(8, 10))
    parts = plt.violinplot(data, positions=[1, 2, 3], showmeans=False, showmedians=True, showextrema=False)

    # Customize each violin body
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    # Customize medians
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')

    plt.xticks([1, 2, 3], labels)
    plt.ylim([ymin, ymax])
    plt.xlabel(maintitle)
    plt.ylabel(ylabel)
    plt.title(maintitle, fontweight='bold')
    plt.tight_layout()
    plt.show()


def sel_objs(objs,season):
    """select objects that cross the North Atlantic, central Arctic and Beaufort region """

    objs_sel = objs.sel_season(Season[season]) \
                                .sel_by_domain(Domains.NORTH_ATLANTIC_JLa.value,type_="anytime",
                                               domain_frac=0.,select_last_timesteps = True) \
                                .sel_by_domain(Domains.CENTRAL_ARCTIC_JLa.value,type_="anytime",
                                               domain_frac=0.,select_last_timesteps = True)  \
                                .sel_by_domain(Domains.BEAUFORT_SIBERIAN_JLa.value,type_="anytime",
                                               domain_frac=0.,select_last_timesteps= True)

    ids_sel=[ds.id_.item() for ds in objs_sel]

    return objs_sel, ids_sel

def sel_central_Arctic(objs,season):

    objs_sel = objs.sel_season(Season[season]) \
                                .sel_by_domain(Domains.CENTRAL_ARCTIC_JLa.value,type_="anytime",
                                               domain_frac=0.,select_last_timesteps = True)
    ids_sel=[ds.id_.item() for ds in objs_sel]

    return objs_sel, ids_sel

def sel_all_objects(objs,season):
    
    objs_sel = objs.sel_season(Season[season])
    
    ids_sel=[ds.id_.item() for ds in objs_sel]

    return objs_sel, ids_sel

def sort_to_dt(dates_list):
    df = pd.DataFrame({"Date": pd.to_datetime(dates_list)})
    df["Year"] = df["Date"].dt.year

    df = df["Year"].value_counts().sort_index().reset_index()
    df.columns = ["Year", "Event Count"]
    df = df
    return df


def sort_to_dt_TA(dates_list):
    df = pd.DataFrame({"Date": pd.to_datetime(dates_list)})
    df["Year"] = df["Date"].dt.year

    df = df["Year"].value_counts().sort_index().reset_index()
    df.columns = ["Year", "WAIs Count"]
    df = df
    return df

def sort_to_dt(dates_list):
    df = pd.DataFrame({"Date": pd.to_datetime(dates_list)})
    df["Year"] = df["Date"].dt.year

    df = df["Year"].value_counts().sort_index().reset_index()
    df.columns = ["Year", "Event Count"]
    df = df
    return df

def split_tracks_180(lons, lats):
    """Splits the tracks at the 180° meridian and add NaNs to cross dateline in plot """
    split_lons, split_lats = [], []

    for i in range(len(lons)):
        split_lons.append(lons[i])
        split_lats.append(lats[i])

        #Add NaN if track is crossing the dateline
        if i < len(lons) - 1 and abs(lons[i + 1] - lons[i]) > 180:
            split_lons.append(np.nan)
            split_lats.append(np.nan)

    return split_lons, split_lats


def transform_coords_to_rot(data_in):

    """Function to transform coordinates into rlat,rlon"""
    
    #Create meshgrid out of standard lat and lon 
    lat_vals = data_in['lat'].values
    lon_vals = data_in['lon'].values
    
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    
    #Transform standard lat and lon into rlat and rlon 
    rlon, rlat = crs_rot.transform_points(ccrs.PlateCarree(), lon_grid, lat_grid)[:, :, 0], \
                 crs_rot.transform_points(ccrs.PlateCarree(), lon_grid, lat_grid)[:, :, 1]
    
    #replace 
    data_out = data_in.assign_coords(rlat=(('lat', 'lon'), rlat), rlon=(('lat', 'lon'), rlon))

    return data_out




######################################################
###############FUNCTIONS FOR ANIMATIONS###############
######################################################

def setup_ax(ax, title):
    ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
    ax.gridlines(linestyle='--', zorder=3)
    ax.coastlines(zorder=2)
    ax.add_feature(cartopy.feature.LAND, color='lightgray', alpha=1, zorder=0, edgecolor='black')
    ax.set_title(title, y=1.05)

    # Clip to polar circle
    center, radius = [0.5, 0.5], 0.55
    circle = mpath.Path(np.vstack([np.sin(np.linspace(0, 2*np.pi, 100)),
                                   np.cos(np.linspace(0, 2*np.pi, 100))]).T * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)


def plot_objects(ax, ds_moaap_sel, df_objs, objs_plot_ids, objs, tt, crs_rot):
    obj_ids_tt = np.unique(ds_moaap_sel.IVT_Objects)
    obj_ids_tt = obj_ids_tt[obj_ids_tt != 0].astype(int)

    for obj_id in obj_ids_tt:
        color = df_objs[df_objs['id'] == str(obj_id)].color.item() if (df_objs['id'] == str(obj_id)).any() else 'b'
        lw = 3 if str(obj_id) in objs_plot_ids else 1

        if str(obj_id) in objs_plot_ids:
            obj_idx = df_objs[df_objs['id'] == str(obj_id)].index.item()
            track = objs[obj_idx].track
            track_lats = np.array([x.lat for x in track.values])
            track_lons = np.array([x.lon for x in track.values])
            track_tt = track.sel(times=tt, method='nearest')
            ax.scatter(track_tt.item().lon, track_tt.item().lat, transform=crs_rot,
                       facecolor=color, edgecolor='k', s=20)
            ax.plot(track_lons, track_lats, color=color, lw=0.75, transform=crs_rot, zorder=8)
            ax.text(track_lons[-1], track_lats[-1], obj_id, transform=crs_rot, color=color, zorder=11)

        (ds_moaap_sel.IVT_Objects == obj_id).plot.contour(x='rlon', y='rlat', ax=ax, levels=1,
                                                          colors=[color], linewidths=lw,
                                                          transform=crs_rot, zorder=10)
# could be modified to a simpler version at the beginning, later use this as base 

# Define the plotting function
def moaap_object_plot(tt,ds_moaap,df_objs,objs_plot_ids,objs,ax=None,save_figs=False,show_figs=True):


    # Time as str
    tt_mod=str(tt.values.astype('datetime64[ns]'))
    start_time= tt - pd.Timedelta(hours=12)
    end_time= tt + pd.Timedelta(hours=12)
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.Orthographic(0, 90)})
        
    ax.clear()  # Clear previous plot

    #fig, ax = plt.subplots(1, 1,subplot_kw={'projection': ccrs.Orthographic(0,90)}) #,figsize=(8,5)

    # Select slice of data that is closest to time tt 
    ds_moaap_sel=ds_moaap.sel(time=tt,method='nearest').compute()

    # Plot the IVT
    ds_moaap_sel.IVT.plot(x='rlon',y='rlat',ax=ax,vmin=0,vmax=250,cmap='Blues',transform=crs_rot,cbar_kwargs={'extend':'both'})#,'pad':0.01})
    
    
    plot_objects(ax, ds_moaap_sel, df_objs, objs_plot_ids, objs, tt, crs_rot)
    
    #tt_mod=str(tt.values.astype('datetime64[ns]'))
    #ax.set_title('')
    #ax.set_title('{}'.format(tt_mod), loc='left')
    
    ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
    ax.gridlines(linestyle='--',zorder=3)
    ax.coastlines(zorder=2)
    ax.add_feature(cartopy.feature.LAND, color='lightgray',alpha=0.8,zorder=0, edgecolor='black') 
    ax.add_feature(cartopy.feature.OCEAN, color='white',zorder=0, edgecolor='None')
    
    ax.set_aspect('equal')
    center, radius = [0.5, 0.5], 0.55
    circle = mpath.Path(np.vstack([np.sin(np.linspace(0, 2*np.pi, 100)), np.cos(np.linspace(0, 2*np.pi, 100))]).T * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_title(f"Time: {pd.to_datetime(tt.values)}",y=1.05)

    if save_figs==True: 
        path_input_images='movie_1140534/'
        path_movies='/work/aa0049/a271122/scripts/movies/'
        plt.tight_layout()
        plt.savefig(path_movies + path_input_images + f"{pd.to_datetime(tt.values).strftime('%Y-%m-%dT%H-%M')}.jpg",dpi=300,transparent=True)
        plt.close(fig)
        
    if show_figs==True:
        plt.tight_layout()
        plt.show()
        
    else:
        plt.clf()
        plt.close(fig)
    return ax


# Plotting function for object with climatic variables 

def clim_moaap_plot(ds,tt,ds_moaap,df_objs,objs_plot_ids,objs,var,var_,cmap='Blues',save_figs=False,show_figs=True):
# def clim_moaap_plot(tt=None,var=None,var_=None,cmap='Blues','enddate=''):

    # Create Figure
    fig, ax = plt.subplots(1, 1,subplot_kw={'projection': ccrs.Orthographic(0,90)}) #,figsize=(8,5)

    # Select and plot the variable
     #_2020010100-2021010100.ncz')
    vardata = ds[var_].sel(time=tt,method='nearest')
    vardata = vardata.assign_coords(lon=((vardata.lon + 180) % 360) - 180).sortby(['lat','lon'])
    vardata.plot(ax=ax,cmap=cmap,transform=ccrs.PlateCarree())
    
    # Select slice of data that is closest to time tt 
    ds_moaap_sel=ds_moaap.sel(time=tt,method='nearest').compute()

    # Plot the tracked objects with individual colors
    obj_ids_tt=np.unique(ds_moaap_sel.IVT_Objects)
    obj_ids_tt = obj_ids_tt[obj_ids_tt != 0].astype(int)
    
    for obj_id in obj_ids_tt:
        if (df_objs['id'] == str(obj_id)).any():
            obj_color=df_objs[df_objs['id'] == str(obj_id)].color.item() #[df_objs.loc[str(obj_id)].color] 
        else:
            obj_color='b'
            
        if str(obj_id) in objs_plot_ids:
            lw=3

            obj_idx=df_objs[df_objs['id'] == str(obj_id)].index.item() #index of obj_id in the df_objs
            track=objs[obj_idx].track
            track_lats = np.array([x.lat for x in track.values])
            track_lons = np.array([x.lon for x in track.values])
            track_tt=track.sel(times=tt,method='nearest')
            ax.scatter(track_tt.item().lon,track_tt.item().lat,transform=crs_rot,facecolor=[obj_color],edgecolor='k',s=20)
            ax.plot(track_lons,track_lats,color=obj_color,lw=0.75,transform=crs_rot,zorder=8) 
            ax.text(track_lons[-1],track_lats[-1],obj_id,transform=crs_rot,color=obj_color,zorder=11)

        else:
            lw=1
                
        (ds_moaap_sel.IVT_Objects==obj_id).plot.contour(x='rlon',y='rlat',ax=ax,levels=1,colors=[obj_color],linewidths=lw,transform=crs_rot,zorder=10)


    for domain_name, domain_data in mask_domains.items():
        domain_data["mask"].plot.contour(levels=1,colors=domain_data["color"],linewidths=1.5,linestyles='--',zorder=4,ax=ax,transform=ccrs.PlateCarree())
   
    ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
    ax.gridlines(linestyle='--',zorder=3)
    ax.coastlines(zorder=2)
    ax.add_feature(cartopy.feature.LAND, color='lightgray',alpha=0.8,zorder=0, edgecolor='black') 
    ax.add_feature(cartopy.feature.OCEAN, color='white',zorder=0, edgecolor='None')
    
    ax.set_aspect('equal')
    center, radius = [0.5, 0.5], 0.55
    circle = mpath.Path(np.vstack([np.sin(np.linspace(0, 2*np.pi, 100)), np.cos(np.linspace(0, 2*np.pi, 100))]).T * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_title(f"Time: {pd.to_datetime(tt.values)}",y=1.05)  
    
    if save_figs==True:
        #output_path = output_dir+'{}.jpg'.format(tt_mod.replace(":", "-"))
        output_path = '/work/aa0049/a271122/scripts/movies/movie_MOAAP_ICON+ERA5_deepWAIs_DJF/{}.jpg'.format(tt_mod.replace(":", "-"))
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show_figs==True:
        plt.show()
    else:
        plt.clf()
        plt.close(fig)


def rotate_vectors(lat, lon, U, V, source_crs=ccrs.PlateCarree(), target_crs=ccrs.PlateCarree()):
    """
    Rotate vector components defined in geographical coordinates (U along east-west, V along north-south)
    into a target projection.

    Parameters
    ----------
    lat : np.ndarray
        2D array of latitudes (degrees north).
    lon : np.ndarray
        2D array of longitudes (degrees east).
    U : np.ndarray
        2D array of vector component in the east-west direction (geographic).
    V : np.ndarray
        2D array of vector component in the north-south direction (geographic).
    source_crs : cartopy.crs.CRS, optional
        The coordinate reference system of the input lat/lon data. Default is PlateCarree.
    target_crs : cartopy.crs.CRS, optional
        The target projection to which we want to rotate the vectors. Default is PlateCarree (no change).

    Returns
    -------
    x : np.ndarray
        2D array of x-coordinates in the target projection.
    y : np.ndarray
        2D array of y-coordinates in the target projection.
    u_rot : np.ndarray
        2D array of rotated U-components aligned with the target projection coordinate system.
    v_rot : np.ndarray
        2D array of rotated V-components aligned with the target projection coordinate system.

    Notes
    -----
    - This function assumes that U and V are defined with respect to geographic east and north.
    - The method involves computing local directions of east and north in the target projection by
      transforming slightly shifted coordinates.
    - No plotting is done inside this function.
    """

    # Transform the given lat/lon points into the target projection
    xyz = target_crs.transform_points(source_crs, lon, lat)
    x, y = xyz[..., 0], xyz[..., 1]

    # Define small increments for direction vector calculation
    delta_lon = 0.01
    delta_lat = 0.01

    # Compute projected coordinates slightly to the east
    xyz_east = target_crs.transform_points(source_crs, lon + delta_lon, lat)
    dx_east = xyz_east[..., 0] - x
    dy_east = xyz_east[..., 1] - y

    # Compute projected coordinates slightly to the north
    xyz_north = target_crs.transform_points(source_crs, lon, lat + delta_lat)
    dx_north = xyz_north[..., 0] - x
    dy_north = xyz_north[..., 1] - y

    # Normalize to get unit direction vectors for east and north in the target projection
    norm_e = np.hypot(dx_east, dy_east)
    e_x = dx_east / norm_e
    e_y = dy_east / norm_e

    norm_n = np.hypot(dx_north, dy_north)
    n_x = dx_north / norm_n
    n_y = dy_north / norm_n

    # Rotate the original (U, V) vectors from geographic frame to target projection frame
    u_rot = U * e_x + V * n_x
    v_rot = U * e_y + V * n_y

    return x, y, u_rot, v_rot

#
def plot_subplot_anomalies_3(tt, variables, variable_titles, variable_names, variable_cmaps, ds_moaap, df_objs, objs, objs_plot_ids, crs_rot, mask_domains=None,savefig=False):
    """
    Create a 2x3 subplot of IVT and other variables at time `tt`.

    Parameters:
        tt: datetime-like object
        variables: list of 6 variable folder names (first one should be 'IVT')
        variable_names: list of 6 variable data names in the NetCDF files
        variable_cmaps: list of 6 colormaps for plotting
        enddate: string, end date for loading variable files
        ds_moaap: main dataset with IVT and object tracking
        df_objs: dataframe with object info (id, color)
        objs: list of object data (with track, speed, etc.)
        objs_plot_ids: list of object ids to highlight
        crs_rot: rotated pole CRS
        mask_domains: dict of domains with masks and colors (optional)
    
    plt.rcParams.update({
    'axes.titlesize': 24,        # Title size
    'axes.labelsize': 20,         # Axis label size
    'lines.linewidth': 3,         # Line width
    'lines.markersize': 10,       # Marker size for lines
    'xtick.labelsize': 16,        # X-tick label size
    'ytick.labelsize': 16         # Y-tick label size
    })
    """

    fig, axs = plt.subplots(1, 3, figsize=(18, 10),
                            subplot_kw={'projection': ccrs.Orthographic(0, 90)})
    axs = axs.flatten()


    for i in range(3):
        ax = axs[i]
        ax.clear()
        ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
        ax.gridlines(linestyle='--', zorder=3)
        ax.coastlines(zorder=2)
        ax.add_feature(cartopy.feature.LAND, color='lightgray', alpha=1, zorder=1, edgecolor='black')
        ax.add_feature(cartopy.feature.OCEAN, color='white', zorder=0, edgecolor='black')
        #fig.suptitle(f'Transarctic WAIs (IVT Objects) and large-scale features for ICON+ERA5 DJF | {pd.to_datetime(tt.values)}',fontweight='bold',y=0.97) #,ha='right' 
        
        if i == 0: 
            ax.set_title("IVT", y=1.09,fontsize=16)
        else: 
            ax.set_title(f"{variable_titles[i]} \n Anomalies", y=1.09,fontsize=16)

        # Clip to circle
        center, radius = [0.5, 0.5], 0.55
        circle = mpath.Path(np.vstack([np.sin(np.linspace(0, 2*np.pi, 100)),
                                       np.cos(np.linspace(0, 2*np.pi, 100))]).T * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

        if i == 0:
            # Plot IVT
            ds_moaap_sel = ds_moaap.sel(time=tt, method='nearest').compute()
            ds_IVT = ds_moaap_sel.IVT.plot(x='rlon', y='rlat', ax=ax, vmin=0, vmax=250,
                                  cmap=variable_cmaps[i], transform=crs_rot,add_colorbar=False)

            cbar = plt.colorbar(ds_IVT, ax=ax, orientation='vertical', pad=0.08, shrink=0.5, extend='both')
            cbar.set_label('IVT [kg m-1 s-1]', fontsize=12)
            cbar.ax.tick_params(labelsize=12)
        else:
            # Plot climate variable
            var = variables[i]
            var_ = variable_names[i]

            # Default padding between colorbar and map
            vmin, vmax = vmin_vmax_dict.get(var, (None, None))
            
            ds = xr.open_mfdataset(
                f'/work/bm1159/XCES/data4xces/reanalysis/reanalysis/ECMWF/IFS/ERA5/1hr/atmos/{var}/r1i1p1/{var}_1hr_reanalysis_era5_r1i1p1_*.nc',chunks={"time":10}
            )
            ds_clim = xr.open_dataset(f'{ERA5_clim_path}/{var}/clim/{var}_clim_day_DJF.nc',chunks={'time': 10})

            #Compute anomaly
            tt_dt = pd.to_datetime(tt.values)
            month = tt_dt.month
            day = tt_dt.day
            month_day_str = f"{month:02d}-{day:02d}"
            
            
            var_tot = ds[var_].sel(time=tt, method='nearest')
            
            var_clim = ds_clim[var_].sel(time=ds_clim.time.dt.strftime('%m-%d') == month_day_str)
            
            var_anom = var_tot - var_clim

            var_anom = var_anom.assign_coords(lon=((var_anom.lon + 180) % 360) - 180).sortby(['lat','lon'])

            var_anom_plot = var_anom.plot(ax=ax, cmap=variable_cmaps[i], transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,add_colorbar=False)

            add_custom_colorbar(var_anom_plot, ax, ds, var_)

    
    # Selec

        # Plot tracked objects
        ds_moaap_sel = ds_moaap.sel(time=tt, method='nearest').compute()
        obj_ids_tt = np.unique(ds_moaap_sel.IVT_Objects)
        obj_ids_tt = obj_ids_tt[obj_ids_tt != 0].astype(int)

        for obj_id in obj_ids_tt:
            if (df_objs['id'] == str(obj_id)).any():
                obj_color = df_objs[df_objs['id'] == str(obj_id)].color.item()
            else:
                obj_color = 'b'

            lw = 3 if str(obj_id) in objs_plot_ids else 1

            if str(obj_id) in objs_plot_ids:
                obj_idx = df_objs[df_objs['id'] == str(obj_id)].index.item()
                track = objs[obj_idx].track
                track_lats = np.array([x.lat for x in track.values])
                track_lons = np.array([x.lon for x in track.values])
                track_tt = track.sel(times=tt, method='nearest')
                ax.scatter(track_tt.item().lon, track_tt.item().lat, transform=crs_rot,
                           facecolor='cyan', edgecolor='k', s=20)
                ax.plot(track_lons, track_lats, color='cyan', lw=0.75, transform=crs_rot, zorder=8)
                ax.text(track_lons[-1], track_lats[-1], obj_id, transform=crs_rot, color='cyan', zorder=11)

            (ds_moaap_sel.IVT_Objects == obj_id).plot.contour(x='rlon', y='rlat', ax=ax, levels=1,
                                                              colors='cyan', linewidths=lw,
                                                              transform=crs_rot, zorder=10)

        #Add sea ice
        #sea_ice_mask.plot.contour(levels=1, ax = ax, transform = ccrs.PlateCarree(), colors = 'white', linewidths = 2, linestyles = '--',zorder=1)


        # Optional domain outlines
        if mask_domains:
            for domain_name, domain_data in mask_domains.items():
                domain_data["mask"].plot.contour(levels=1, colors=domain_data["color"],
                                                 linewidths=1.5, linestyles='--',
                                                 zorder=4, ax=ax, transform=ccrs.PlateCarree())

    if savefig==True:
        path_input_images='subplot_2/'
        path_movies='/work/aa0049/a271122/scripts/movies/'
        plt.savefig(path_movies + path_input_images + f"{pd.to_datetime(tt.values).strftime('%Y-%m-%dT%H-%M')}.jpg",dpi=300)#,transparent=True)
        #plt.close(fig)
        
    #plt.tight_layout()
    plt.show()




def plot_subplot_anomalies_new(tt,dataset, variables, variable_names, variable_cmaps, ds_moaap, df_objs, objs, objs_plot_ids, crs_rot,ds_contour, mask_domains=None,savefig=False):
    """
    Create a 2x3 subplot of IVT and other variables at time `tt`.

    Parameters:
        tt: datetime-like object
        variables: list of 6 variable folder names (first one should be 'IVT')
        variable_names: list of 6 variable data names in the NetCDF files
        variable_cmaps: list of 6 colormaps for plotting
        enddate: string, end date for loading variable files
        ds_moaap: main dataset with IVT and object tracking
        df_objs: dataframe with object info (id, color)
        objs: list of object data (with track, speed, etc.)
        objs_plot_ids: list of object ids to highlight
        crs_rot: rotated pole CRS
        mask_domains: dict of domains with masks and colors (optional)
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 10),
                            subplot_kw={'projection': ccrs.Orthographic(0, 90)})
    axs = axs.flatten()


    for i in range(6):
        ax = axs[i]
        ax.clear()
        ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
        ax.gridlines(linestyle='--', zorder=3)
        ax.coastlines(zorder=2)
        ax.add_feature(cartopy.feature.LAND, color='lightgray', alpha=1, zorder=0, edgecolor='black')
        #ax.add_feature(cartopy.feature.OCEAN, color='white', zorder=0, edgecolor='black')
        fig.suptitle(f'Transarctic WAI (IVT Object) on {pd.to_datetime(tt.values)} and large-scale features in ERA5',fontweight='bold',y=0.97) #,ha='right' 

        if i == 0: 
            ax.set_title("Integrated Vapor Transport \n", y=1.05)
        else: 
            ax.set_title(f"{variable_names[i]} Anomalies", y=1.05)

        # Clip to circle
        center, radius = [0.5, 0.5], 0.55
        circle = mpath.Path(np.vstack([np.sin(np.linspace(0, 2*np.pi, 100)),
                                       np.cos(np.linspace(0, 2*np.pi, 100))]).T * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

        if i == 0:
            # Plot IVT
            ds_moaap_sel = ds_moaap.sel(time=tt, method='nearest').compute()
            ds_IVT = ds_moaap_sel.IVT.plot(x='rlon', y='rlat', ax=ax, vmin=0, vmax=250,
                                  cmap=variable_cmaps[i], transform=crs_rot,add_colorbar=False)
            #ax.quiver(lon_plot, lat_plot, u_plot, v_plot,transform=ccrs.PlateCarree(), color='k', scale=500, zorder=3)

            cbar = plt.colorbar(ds_IVT, ax=ax, orientation='vertical', pad=0.08, shrink=0.8, extend='both')
            cbar.set_label('IVT (kg m-1 s-1)', fontsize=10)
            cbar.ax.tick_params(labelsize=10)
        else:
            # Plot climate variable
            var = variables[i]

            # Default padding between colorbar and map
            vmin, vmax = vmin_vmax_dict.get(var, (None, None))

            ds = dataset[i]
            ds_clim = xr.open_dataset(f'{ERA5_clim_path}/{var}/clim/{var}_clim_day_DJF.nc',chunks={'time': 10})

            #Compute anomaly
            tt_dt = pd.to_datetime(tt.values)
            month = tt_dt.month
            day = tt_dt.day
            month_day_str = f"{month:02d}-{day:02d}"
            
            
            var_tot = ds[var].sel(time=tt, method='nearest')


            var_clim = ds_clim[var].sel(time=ds_clim.time.dt.strftime('%m-%d') == month_day_str)
            
            var_anom = var_tot - var_clim
            var_anom = var_anom.assign_coords(lon=((var_anom.lon + 180) % 360) - 180).sortby(['lat','lon'])
            
            var_anom_plot = var_anom.plot(ax=ax, cmap=variable_cmaps[i], transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,add_colorbar=False)

            #add GPH contour
            contour = ds_contour.plot.contour(levels=10,ax=ax,transform=ccrs.PlateCarree(),colors='k',linewidths=0.5)
            ax.clabel(contour, fmt='%1.0f', inline=True, fontsize=7)
        
            add_custom_colorbar(var_anom_plot, ax, ds, var)

        # Plot tracked objects
        ds_moaap_sel = ds_moaap.sel(time=tt, method='nearest').compute()
        obj_ids_tt = np.unique(ds_moaap_sel.IVT_Objects)
        obj_ids_tt = obj_ids_tt[obj_ids_tt != 0].astype(int)

        for obj_id in obj_ids_tt:
            if (df_objs['id'] == str(obj_id)).any():
                obj_color = df_objs[df_objs['id'] == str(obj_id)].color.item()
            else:
                obj_color = 'b'

            lw = 3 if str(obj_id) in objs_plot_ids else 1

            if str(obj_id) in objs_plot_ids:
                obj_idx = df_objs[df_objs['id'] == str(obj_id)].index.item()
                track = objs[obj_idx].track
                track_lats = np.array([x.lat for x in track.values])
                track_lons = np.array([x.lon for x in track.values])
                track_tt = track.sel(times=tt, method='nearest')
                ax.scatter(track_tt.item().lon, track_tt.item().lat, transform=crs_rot,
                           facecolor=[obj_color], edgecolor='k', s=20)
                ax.plot(track_lons, track_lats, color=obj_color, lw=0.75, transform=crs_rot, zorder=8)
                ax.text(track_lons[-1], track_lats[-1], obj_id, transform=crs_rot, color=obj_color, zorder=11)

            (ds_moaap_sel.IVT_Objects == obj_id).plot.contour(x='rlon', y='rlat', ax=ax, levels=1,
                                                              colors=[obj_color], linewidths=lw,
                                                              transform=crs_rot, zorder=10)

        #Add sea ice
        ds_sic_DJF.sic.plot.contour(levels=1, ax = ax, transform = ccrs.PlateCarree(), colors = 'cyan', linewidths = 1, linestyles = '-',zorder=1)


        # Optional domain outlines
        if mask_domains:
            for domain_name, domain_data in mask_domains.items():
                domain_data["mask"].plot.contour(levels=1, colors=domain_data["color"],
                                                 linewidths=1.5, linestyles='--',
                                                 zorder=4, ax=ax, transform=ccrs.PlateCarree())

        
    if savefig==True:
        plt.savefig('/work/aa0049/a271122/scripts/'+f"casestudy_2012_{variables[0]}.jpg",dpi=300)#,transparent=True)
        #plt.close(fig)
        
    #plt.tight_layout()
    plt.show()


def plot_subplot_anomalies_noIVT(tt,dataset, variables, variable_names, variable_cmaps, ds_moaap, df_objs, objs, objs_plot_ids, crs_rot,ds_contour, mask_domains=None,savefig=False):
    """
    Create a 2x3 subplot of IVT and other variables at time `tt`.

    Parameters:
        tt: datetime-like object
        variables: list of 6 variable folder names (first one should be 'IVT')
        variable_names: list of 6 variable data names in the NetCDF files
        variable_cmaps: list of 6 colormaps for plotting
        enddate: string, end date for loading variable files
        ds_moaap: main dataset with IVT and object tracking
        df_objs: dataframe with object info (id, color)
        objs: list of object data (with track, speed, etc.)
        objs_plot_ids: list of object ids to highlight
        crs_rot: rotated pole CRS
        mask_domains: dict of domains with masks and colors (optional)
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 10),
                            subplot_kw={'projection': ccrs.Orthographic(0, 90)})
    axs = axs.flatten()



    for i in range(6):
        ax = axs[i]
        ax.clear()
        ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
        ax.gridlines(linestyle='--', zorder=3)
        ax.coastlines(zorder=2)
        ax.add_feature(cartopy.feature.LAND, color='lightgray', alpha=1, zorder=0, edgecolor='black')
        #ax.add_feature(cartopy.feature.OCEAN, color='white', zorder=0, edgecolor='black')
        #fig.suptitle(' ',fontweight='bold',y=0.97) #,ha='right'

        # Clip to circle
        center, radius = [0.5, 0.5], 0.55
        circle = mpath.Path(np.vstack([np.sin(np.linspace(0, 2*np.pi, 100)),
                                       np.cos(np.linspace(0, 2*np.pi, 100))]).T * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

       
        # Plot climate variable
        var = variables[i]

        # Default padding between colorbar and map
        vmin, vmax = vmin_vmax_dict.get(var, (None, None))

        ds = dataset[i]
        ds_clim = xr.open_dataset(f'{ERA5_clim_path}/{var}/clim/{var}_clim_day_DJF.nc',chunks={'time': 10})

        #Compute anomaly
        tt_dt = pd.to_datetime(tt.values)
        month = tt_dt.month
        day = tt_dt.day
        month_day_str = f"{month:02d}-{day:02d}"

        if var =='pr':
            var_tot = ds[var].sel(time=tt, method='nearest')*3600

            var_clim = ds_clim[var].sel(time=ds_clim.time.dt.strftime('%m-%d') == month_day_str)*3600
        else: 
            var_tot = ds[var].sel(time=tt, method='nearest')
    
            var_clim = ds_clim[var].sel(time=ds_clim.time.dt.strftime('%m-%d') == month_day_str)
            
        var_anom = var_tot - var_clim
        var_anom = var_anom.assign_coords(lon=((var_anom.lon + 180) % 360) - 180).sortby(['lat','lon'])

        var_anom_plot = var_anom.plot(ax=ax, cmap=variable_cmaps[i], transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,add_colorbar=False)

        #add GPH contour
        contour = ds_contour.plot.contour(levels=10,ax=ax,transform=ccrs.PlateCarree(),colors='k',linewidths=0.5)
        ax.clabel(contour, fmt='%1.0f', inline=True, fontsize=7)
    
        add_custom_colorbar(var_anom_plot, ax, ds, var)

        # Plot tracked objects
        ds_moaap_sel = ds_moaap.sel(time=tt, method='nearest').compute()
        obj_ids_tt = np.unique(ds_moaap_sel.IVT_Objects)
        obj_ids_tt = obj_ids_tt[obj_ids_tt != 0].astype(int)

        for obj_id in obj_ids_tt:
            if (df_objs['id'] == str(obj_id)).any():
                obj_color = df_objs[df_objs['id'] == str(obj_id)].color.item()
            else:
                obj_color = 'b'

            lw = 3 if str(obj_id) in objs_plot_ids else 1

            if str(obj_id) in objs_plot_ids:
                obj_idx = df_objs[df_objs['id'] == str(obj_id)].index.item()
                track = objs[obj_idx].track
                track_lats = np.array([x.lat for x in track.values])
                track_lons = np.array([x.lon for x in track.values])
                track_tt = track.sel(times=tt, method='nearest')
                ax.scatter(track_tt.item().lon, track_tt.item().lat, transform=crs_rot,
                           facecolor=[obj_color], edgecolor='k', s=20)
                ax.plot(track_lons, track_lats, color=obj_color, lw=0.75, transform=crs_rot, zorder=8)
                ax.text(track_lons[-1], track_lats[-1], obj_id, transform=crs_rot, color=obj_color, zorder=11)

            (ds_moaap_sel.IVT_Objects == obj_id).plot.contour(x='rlon', y='rlat', ax=ax, levels=1,
                                                              colors=[obj_color], linewidths=lw,
                                                              transform=crs_rot, zorder=10)

        #Add sea ice
        ds_sic_DJF.sic.plot.contour(levels=1, ax = ax, transform = ccrs.PlateCarree(), colors = 'cyan', linewidths = 1, linestyles = '-',zorder=1)


        # Optional domain outlines
        if mask_domains:
            for domain_name, domain_data in mask_domains.items():
                domain_data["mask"].plot.contour(levels=1, colors=domain_data["color"],
                                                 linewidths=1.5, linestyles='--',
                                                 zorder=4, ax=ax, transform=ccrs.PlateCarree())

        ax.set_title(f"{variable_names[i]} Anomalies", y=1.05)
        #ax.set_title(f"{variable_names[i]} Anomalies", y=1.05)
        
        
    plt.subplots_adjust(wspace=0.01)
        
    if savefig==True:
        plt.savefig("/work/aa0049/a271122/scripts/ERA5/casestudy_pr.jpg",dpi=300,bbox_inches="tight")#,transparent=True)
        #plt.close(fig)
        
    #plt.tight_layout()
    plt.show()

def plot_subplot_anomalies(tt,dataset, variables, variable_names,
                           variable_cmaps, ds_moaap, df_objs, objs_final, objs_plot_ids,
                           crs_rot,ds_contour, mask_domains=None,savefig=False):
    """
    Create a 2x3 subplot of IVT and other variables at time tt.

    Parameters:
        tt: datetime-like object
        variables: list of 6 variable folder names (first one should be 'IVT')
        variable_names: list of 6 variable data names in the NetCDF files
        variable_cmaps: list of 6 colormaps for plotting
        enddate: string, end date for loading variable files
        ds_moaap: main dataset with IVT and object tracking
        df_objs: dataframe with object info (id, color)
        objs: list of object data (with track, speed, etc.)
        objs_plot_ids: list of object ids to highlight
        crs_rot: rotated pole CRS
        mask_domains: dict of domains with masks and colors (optional)
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 14),
                            subplot_kw={'projection': ccrs.Orthographic(0, 90)})
    axs = axs.flatten()

    for i in range(0,6):
        ax = axs[i]
        ax.clear()
        ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
        ax.gridlines(linestyle='--', zorder=3)
        ax.coastlines(zorder=2)
        ax.add_feature(cartopy.feature.LAND, color='lightgray', alpha=1, zorder=0, edgecolor='black')
        #ax.add_feature(cartopy.feature.OCEAN, color='white', zorder=0, edgecolor='black')
        #fig.suptitle(f'Transarctic WAIs (IVT Objects) and large-scale features for ERA5 DJF | {pd.to_datetime(tt.values)}',fontweight='bold',y=0.97) #,ha='right' 

        
        #Clip to circle
        center, radius = [0.5, 0.5], 0.55
        circle = mpath.Path(np.vstack([np.sin(np.linspace(0, 2*np.pi, 100)),
                                       np.cos(np.linspace(0, 2*np.pi, 100))]).T * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

        #Add GPH contour
        contour = ds_contour.plot.contour(levels=10,ax=ax,transform=ccrs.PlateCarree(),colors='k',linewidths=0.5)
        ax.clabel(contour, fmt='%1.0f', inline=True, fontsize=7)

        # Optional domain outlines
        if mask_domains:
            for domain_name, domain_data in mask_domains.items():
                domain_data["mask"].plot.contour(levels=1, colors=domain_data["color"],
                                                 linewidths=1.5, linestyles='--',zorder=4, ax=ax, transform=ccrs.PlateCarree())


        # Plot climate variable
        var = variables[i]

        # Default padding between colorbar and map
        vmin, vmax = vmin_vmax_dict.get(var, (None, None))

        ds = dataset[i]
        ds_clim = xr.open_dataset(f'{ERA5_clim_path}/{var}/clim/{var}_clim_day_DJF.nc',chunks={'time': 10})

        #Compute anomaly
        tt_dt = pd.to_datetime(tt.values)
        month = tt_dt.month
        day = tt_dt.day
        month_day_str = f"{month:02d}-{day:02d}"
        
        
        var_tot = ds[var].sel(time=tt, method='nearest')


        var_clim = ds_clim[var].sel(time=ds_clim.time.dt.strftime('%m-%d') == month_day_str)
        
        var_anom = var_tot - var_clim
        var_anom = var_anom.assign_coords(lon=((var_anom.lon + 180) % 360) - 180).sortby(['lat','lon'])
        
        var_anom_plot = var_anom.plot(ax=ax, cmap=variable_cmaps[i], transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,add_colorbar=False)

        
        add_custom_colorbar(var_anom_plot, ax, ds, var)

        # Plot tracked objects
        ds_moaap_sel = ds_moaap.sel(time=tt, method='nearest').compute()
        obj_ids_tt = np.unique(ds_moaap_sel.IVT_Objects)
        obj_ids_tt = obj_ids_tt[obj_ids_tt != 0].astype(int)

        for obj_id in obj_ids_tt:
            if (df_objs['id'] == str(obj_id)).any():
                obj_color = 'plum' #df_objs[df_objs['id'] == str(obj_id)].color.item()
            else:
                obj_color = 'b'

            lw = 3.5 if str(obj_id) in objs_plot_ids else 1

            if str(obj_id) in objs_plot_ids:
                obj_idx = df_objs[df_objs['id'] == str(obj_id)].index.item()
                track = objs_final[obj_idx].track
                track_lats = np.array([x.lat for x in track.values])
                track_lons = np.array([x.lon for x in track.values])
                track_tt = track.sel(times=tt, method='nearest')
                ax.scatter(track_tt.item().lon, track_tt.item().lat, transform=crs_rot,
                           facecolor=[obj_color], edgecolor='k', s=20)
                ax.plot(track_lons, track_lats, color=obj_color, lw=1.25, transform=crs_rot, zorder=8)
                ax.text(track_lons[-1], track_lats[-1], obj_id, transform=crs_rot, color=obj_color, zorder=11)

            (ds_moaap_sel.IVT_Objects == obj_id).plot.contour(x='rlon', y='rlat', ax=ax, levels=1,
                                                              colors=[obj_color], linewidths=lw,
                                                              transform=crs_rot, zorder=10)

        #Add sea ice
        ds_sic_DJF.sic.plot.contour(levels=1, ax = ax, transform = ccrs.PlateCarree(), colors = 'cyan', linewidths = 1, linestyles = '-',zorder=1)


        ax.set_title(f"{variable_names[i]} Anomalies", y=1.07)
        
    plt.subplots_adjust(wspace=0.01)
    #plt.subplots_adjust(wspace=0.01)
    #plt.tight_layout()
    
    if savefig==True:
        path_input_images='ERA5/'
        path_movies='/work/aa0049/a271122/scripts/'
        plt.savefig(path_movies + path_input_images + f"casestudy_2_{variables[0]}.jpg",dpi=300,bbox_inches="tight")#,transparent=True)
        #plt.close(fig)



    plt.show()

def add_custom_colorbar(mappable, ax, ds, var_):

    standard_name = ds[var_].attrs.get('standard_name', var_)
    if var_=='pr':
        units = 'mm h-1'
    else:
        units = ds[var_].attrs.get('units', '')

    cbar = plt.colorbar(mappable, ax=ax, orientation='horizontal', pad=0.08, shrink = 0.8, extend='both')

       # force at least 5 ticks
    cbar.locator = MaxNLocator(nbins=5, min_n_ticks=5)
    cbar.update_ticks()
    
    cbar.set_label(f'{var_} ({units})', fontsize=15)
    cbar.ax.tick_params(labelsize=12)





