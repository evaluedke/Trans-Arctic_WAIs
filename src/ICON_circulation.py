# Contains functions for large-scale circulation regime analysis in ICON-CLM
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
from collections import defaultdict


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

domain_hatch = {
    "NA": "---",   # Diagonal stripes
    "CA": "///",    # Vertical stripes
    "BS": "...",     # No pattern (solid fill)
    "NONE": ""   # Optional: Cross-hatch for no domain
}
domain_color = {"NA": "white", "CA": "white", "BS": "white", "NONE": "white"}


legend_patches = [
    mpatches.Patch(facecolor="lightgray", hatch=domain_hatch["NA"], label="North Atlantic"),
    mpatches.Patch(facecolor="lightgray", hatch=domain_hatch["CA"], label="Central Arctic"),
    mpatches.Patch(facecolor="lightgray", hatch=domain_hatch["BS"], label="Beaufort Sea"),
]

arctic_cluster_path = '/work/aa0049/a271122/ERA5/CirculationRegime_clusters/'

#Define regime colors and images in dictionary
atlantic_regimes = {
    '1': {'name':'WINTER-NAO+','color': '#FB9A99', 'image': f'{arctic_cluster_path}/regime_images/djf_atlantic_cluster_1_upper_half_horizontal.png'},
    '2': {'name':'WINTER-SCAN','color': '#A6CEE3', 'image': f'{arctic_cluster_path}/regime_images/djf_atlantic_cluster_2_upper_half_horizontal.png'},
    '3': {'name':'WINTER-ATL-','color': 'navy', 'image': f'{arctic_cluster_path}/regime_images/djf_atlantic_cluster_3_upper_half_horizontal.png'},
    '4': {'name':'WINTER-NAO-','color': '#CAB2D6', 'image': f'{arctic_cluster_path}/regime_images/djf_atlantic_cluster_4_upper_half_horizontal.png'},
    '5': {'name':'WINTER-DIP','color': '#B2DF8A', 'image': f'{arctic_cluster_path}/regime_images/djf_atlantic_cluster_5_upper_half_horizontal.png'}
}

###############################################################
#########################FUNCTIONS#############################
###############################################################


def plot_circulation(df,objs_final,maxn,years,regimes): 
    '''Function to visualize the circulation regime in each object 
    '''
    fig, axes = plt.subplots(maxn, 1, figsize=(6, 9))
    axes_flat = axes.flatten()
    plt.subplots_adjust(hspace=2.6)

    #Plot the sequence of regimes for each object
    for i, obj in enumerate(objs_final[0:maxn]):
        ax = axes_flat[i]
        
        #Object ID 
        obj_id = obj.id_.item()
    
        #Get dates (only once) in which object exists in 2 formats 
        obj_times = pd.to_datetime(obj.times.dt.strftime('%Y-%m-%d %H:%M:%S').values)
        obj_times = obj_times[obj_times.hour == 12]

    
        for j, date in enumerate(obj_times):
            #Extract regime and color for the current date
            regime = df['cluster_id'][df['time'] == date].item()        
            regime_color = regimes[f'{regime}']['color']
    
            #Extract domain information
            domain = df.loc[df['time'] == date, 'domain'].values[0]
    
            #Plot regime and domain
            #Regime in color
            ax.bar(j, 1, width=0.85, color=regime_color, align='center', edgecolor='none')
            #Domain through stripes/pattern
            ax.bar(j, 1, width=0.85, color='none', align='center', 
                   edgecolor=domain_color[domain], hatch=domain_hatch[domain], linewidth=1.5)
            
        #Labels and title
        ax.set_title(f"ID: {obj_id}", fontsize=8,loc = 'left')
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    
    # Add images as legend

    image_positions = [0.7, 0.55, 0.40, 0.25, 0.1]  # Vertical
    width = 0.3
    height = 0.3

    for i, (name, info) in enumerate(regimes.items()):
        img = mpimg.imread(info['image'])
        
        #positions
        ax_img = fig.add_axes([1.0, image_positions[i], width, height])  #[left, bottom, width, height]
        ax_img.imshow(img)
        ax_img.axis('off')
        
        #Add frame and regime title
        rect = Rectangle((0, 0), 1.01, 1.01, linewidth=5, edgecolor=info['color'], facecolor='none', transform=ax_img.transAxes)
        ax_img.add_patch(rect)
        circulation_name = info['name']
        ax_img.text(0.5, 1.05, f'{circulation_name}', ha='center', va='bottom', fontsize=10, color='k', transform=ax_img.transAxes)
    #Add second legend to define domain patterns
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.1,30), ncol=3)
    
    #Add title
    fig.text(0.3,0.98,f'Circulation Regimes for WAIs ({years})',fontsize=15)
    #plt.savefig('/work/aa0049/a271122/ERA5/CirculationRegime_clusters/atlantic_circulation_regimes.png', bbox_inches='tight', dpi=300)
    plt.show()



def get_calendar_days(NA_times,CA_times,BS_times):
    """Function that returns combined calender dates of combined intrusions in different sectors"""
    
    NA_month_day = list(set((time.month, time.day) for time in pd.to_datetime(NA_times)))
    CA_month_day = list(set((time.month, time.day) for time in pd.to_datetime(CA_times)))
    BS_month_day = list(set((time.month, time.day) for time in pd.to_datetime(BS_times)))

    return NA_month_day, CA_month_day, BS_month_day 



def compute_cluster_counts(df, clusters):
    """ Function to compute cluster percentages"""
    return [(len(df[df['cluster_id'] == c]) / len(df) * 100) if len(df) > 0 else 0 for c in clusters]



def plot_cluster_distribution(df, regimes, title,bar_colors,Arctic= False):
    """Funtion to compute clusters as histogram"""
    fig, axs = plt.subplots(2, 2, figsize=(8, 10))
    axs = axs.flatten()

    if Arctic:
        clusters =[1, 2, 3, 4, 5, 6]
    else: 
        clusters = [0,1, 2, 3, 4]
    domains = ["Total", "North Atlantic", "Central Arctic", "Beaufort/Siberian Sea"]
    
    # Compute counts for all regions
    counts_all = compute_cluster_counts(df, clusters)
    counts_NA = compute_cluster_counts(df[df['domain'] == 'NA'], clusters)
    counts_CA = compute_cluster_counts(df[df['domain'] == 'CA'], clusters)
    counts_BS = compute_cluster_counts(df[df['domain'] == 'BS'], clusters)
    
    # List of all counts
    all_counts = [counts_all, counts_NA, counts_CA, counts_BS]
    
    # Plot in each subplot
    for ax, counts, domain in zip(axs, all_counts, domains):
        ax.bar(clusters, counts, color=bar_colors, edgecolor='k', width=0.7,zorder=2)
        ax.set_ylim(0,101)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Percentage [%]')
        ax.set_title(f"{domain}")
        ax.yaxis.grid(True, linestyle="--", alpha=0.7,zorder=1)
        
    # Add global title
    fig.suptitle(f'Distribution of Atlantic Circulation Clusters ({title})', y=0.95,fontsize=14)
     # Add images as legend

    image_positions = [0.67, 0.51, 0.35, 0.19, 0.03]  # Vertical
    width = 0.3
    height = 0.3

    for i, (name, info) in enumerate(regimes.items()):
        img = mpimg.imread(info['image'])
        
        #positions
        ax_img = fig.add_axes([1.0, image_positions[i], width, height])  #[left, bottom, width, height]
        ax_img.imshow(img)
        ax_img.axis('off')
        
        #Add frame and regime title
        rect = Rectangle((0, 0), 1.01, 1.01, linewidth=9, edgecolor=info['color'], facecolor='none', transform=ax_img.transAxes)
        ax_img.add_patch(rect)
        circulation_name=info['name']
        ax_img.text(0.5, 1.05, f'{circulation_name}', ha='center', va='bottom', fontsize=12, color='k', transform=ax_img.transAxes)

    #plt.savefig('/work/aa0049/a271122/ERA5/CirculationRegime_clusters/distriburion_atlantic_clusters.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    plt.tight_layout()
    #plt.savefig('/work/aa0049/a271122/ERA5/CirculationRegime_clusters/distriburion_atlantic_clusters.png', bbox_inches='tight', dpi=300)
    plt.show()



def new_barplots():
    #Plot the sequence of regimes for each object
    for i, obj in enumerate(objs_final[0:12]):
        
        #Object ID 
        obj_id = obj.id_.item()
    
        #Get dates (only once) in which object exists in 2 formats 
        obj_times = pd.to_datetime(obj.times.dt.strftime('%Y-%m-%d %H:%M:%S').values)
        obj_times_6h = obj_times[obj_times.hour.isin([0,6,12,18])]

        for j, date in enumerate(obj_times_6h):
            #Extract regime and color for the current date
            regime = df['cluster'][df['date'] == date].item()        
            regime_color = regimes[f'{regime}']['color']
    
            #Extract domain information
            domain = df.loc[df['date'] == date, 'domain'].values[0]


def create_df_cluster_info(obj_final, NA_times, CA_times, BS_times):
    """Function to return a dataframe containing relevant information on
            date | cluster id | cluster name | domain"""
    
    #Extract dates and cluster info for control period 

    date = []
    cluster_name = []
    for i in range(len(obj_final)): 
        obj = obj_final[i]
        # Create List with all dates
        for j in range(len(obj.times)):
            date.append(pd.to_datetime(obj.times[j].values).strftime('%Y-%m-%dT%H:%M'))
        
            #Create List with all clusters
            cluster_name.append(obj.clusters[j].values.item())


    #Create a dataframe out of MOAAP dates and cluster info 
    
    data_frame = {'date':date,
                    'cluster_names':cluster_name}
    
    df_data = pd.DataFrame(data_frame)
    df_data['date'] = pd.to_datetime(df_data['date'])

    
    # Check domain for each date
    NA_bool = df_data['date'].isin(NA_times)
    CA_bool = df_data['date'].isin(CA_times)
    BS_bool = df_data['date'].isin(BS_times)

    #Create new column for domain
    conditions = [NA_bool, CA_bool, BS_bool]
    choices = ['NA', 'CA', 'BS']
    
    df_data['domain'] = np.select(conditions, choices, default='NONE')
    
    
    #Create new column for cluster ids
    DIP = df_data['cluster_names'] == 'WINTER-DIP'
    NAOplus = df_data['cluster_names'] == 'WINTER-NAO+'
    NAOminus = df_data['cluster_names'] == 'WINTER-NAO-'
    SCAN = df_data['cluster_names'] == 'WINTER-SCAN'
    ATLminus = df_data['cluster_names'] == 'WINTER-ATL-'
    
    conditions = [NAOplus,SCAN,ATLminus,NAOminus,DIP]
    choices = [1,2,3,4,5]
    
    df_data['cluster_id']=np.select(conditions,choices,default='NONE')


    return df_data

def get_normalized_frequency(df_data):
    """Function to get normalized cluster frequency as input for histograms"""
    # Define all cluster numbers from your regime definitions
    all_cluster_nums = [int(k) for k in atlantic_regimes.keys()]
    
    # Initialize total counters per domain and cluster with all clusters preset to 0.0
    cluster_weights = {
        'NA': defaultdict(float, {cl: 0.0 for cl in all_cluster_nums}),
        'CA': defaultdict(float, {cl: 0.0 for cl in all_cluster_nums}),
        'BS': defaultdict(float, {cl: 0.0 for cl in all_cluster_nums}),
    }
    
    
    # Track clusters per domain for this object
    domain_clusters = {'NA': [], 'CA': [], 'BS': []}
    
    for date in df_data['date']:
        row = df_data[df_data['date'] == date]
        domain = row['domain'].values[0]
        cluster = int(row['cluster_id'].values[0])
        if domain in domain_clusters:
            domain_clusters[domain].append(cluster)
    
    # Normalize and add weights
    for domain, clusters in domain_clusters.items():
        if clusters:
            counts = pd.Series(clusters).value_counts(normalize=True)
            for cluster_num in all_cluster_nums:
                cluster_weights[domain][cluster_num] += counts.get(cluster_num, 0.0)
    
    
    # Create mapping from cluster number to cluster name
    cluster_num_to_name = (
        df_data[['cluster_id', 'cluster_names']]
        .drop_duplicates()
        .set_index('cluster_id')['cluster_names']
        .to_dict()
    )
    
    # Convert weights to named version (ensuring all clusters are included)
    named_cluster_weights_total = {
        domain: {
            cluster_num_to_name.get(k, f"Cluster-{k}"): v
            for k, v in sorted(clusters.items())
        }
        for domain, clusters in cluster_weights.items()
    }
    
    
    # iF WE WANT PERCENTAGES
    cluster_num_to_name = {
        int(k): v['name'] for k, v in atlantic_regimes.items()
    }
    
    # Normalize per domain so weights sum to 100%
    named_cluster_weights = {}
    for domain, cluster_dict in cluster_weights.items():
        total_weight = sum(cluster_dict.values())
        if total_weight == 0:
            # Avoid division by zero: assign 0 to all
            normalized = {cluster_num_to_name[k]: 0.0 for k in cluster_num_to_name}
        else:
            normalized = {
                cluster_num_to_name[k]: (v / total_weight) * 100
                for k, v in cluster_dict.items()
            }
    
        # Make sure all cluster names are present, including 0.0 entries
        for k in cluster_num_to_name:
            if cluster_num_to_name[k] not in normalized:
                normalized[cluster_num_to_name[k]] = 0.0
    
        named_cluster_weights[domain] = normalized
    
    return named_cluster_weights
    

def plot_hourly_hist(named_cluster_weights,counts_all,name,NorESM=False): 
    """Function to plot histograms with normalized frequency"""
    bar_colors = ['#FB9A99','#A6CEE3', 'navy', '#CAB2D6','#B2DF8A']
    regime_names = ['NAO+', 'SCAN', 'ATL-', 'NAO-', 'DIP']
    domain_names = {'NA':"North Atlantic",'CA': "Central Arctic", 'BS':"Beaufort/Siberian Sea"}


    fig, axs = plt.subplots(1, 3, figsize=(14, 6))
    axs = axs.flatten()
    
    # Plot in each subplot
    for ax, (domain, weights) in zip(axs,named_cluster_weights.items()):
        clusters = list(weights.keys())
        values = list(weights.values())
    
        
    
        ax.bar(clusters, values, color=bar_colors, edgecolor='k', width=0.7,zorder=2)
        ax.set_ylim(0, 70)
        ax.set_ylabel('Normalized Frequency (%)')
        ax.set_title(f"{domain_names[domain]}")
        ax.set_xticks(clusters)
        ax.set_xticklabels(regime_names)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7,zorder=1)
    
        #Add Climatology clusters
        ax.bar(clusters, counts_all, color='grey', edgecolor='grey', width=0.5,align= 'edge',zorder=1,alpha=0.3)
    
    if NorESM: 
        plt.savefig(f'/work/aa0049/a271122/ERA5/CirculationRegime_clusters/ICON_NorESM_distribution_atlantic_clusters_{name}.png', bbox_inches='tight', dpi=300)
    else: 
        plt.savefig(f'/work/aa0049/a271122/ERA5/CirculationRegime_clusters/ICON_CNRM_distribution_atlantic_clusters_{name}.png', bbox_inches='tight', dpi=300)
    plt.show()

    