from ERA5_utils import *
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.path as mpath
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import scipy
import datetime

from collections import defaultdict
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

import concurrent.futures


### Load MOAAP functions
import sys
sys.path.append('/work/aa0049/a271122/MOAAP/MOAAP/')
from src.utils import *
from src.Corrections import * 
from src.Enumerations import Month, Season, Experiments, Domains
from src.xarray_util import create_obj_from_dict,  ObjectContainer,  load_tracking_objects
from src.plot_funcs import plot_contourf_rotated_grid #, plot_unstructured_rotated_grid

from src.GridPoints import Domain



##########Define Stuff########################
arctic_cluster_path = '/work/aa0049/a271122/ERA5/CirculationRegime_clusters/'

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

#Define regime colors and images in dictionary
atlantic_regimes = {
    '1': {'name':'WINTER-NAO+','color': '#FB9A99',
          'image': f'{arctic_cluster_path}/regime_images/djf_atlantic_cluster_1_upper_half_horizontal.png'},
    '2': {'name':'WINTER-SCAN','color': '#A6CEE3',
          'image': f'{arctic_cluster_path}/regime_images/djf_atlantic_cluster_2_upper_half_horizontal.png'},
    '3': {'name':'WINTER-ATL-','color': 'navy',
          'image': f'{arctic_cluster_path}/regime_images/djf_atlantic_cluster_3_upper_half_horizontal.png'},
    '4': {'name':'WINTER-NAO-','color': '#CAB2D6', 
          'image': f'{arctic_cluster_path}/regime_images/djf_atlantic_cluster_4_upper_half_horizontal.png'},
    '5': {'name':'WINTER-DIP','color': '#B2DF8A',
          'image': f'{arctic_cluster_path}/regime_images/djf_atlantic_cluster_5_upper_half_horizontal.png'}
}


##############################################
##########Define Functions####################
##############################################


def extract_hemisphere_times(ds_obj,PC=False):
    timesList = []
    for time in ds_obj['times']:
        lat = ds_obj.geo_lat.sel(times=time.values).item()
        lon = ds_obj.geo_lon.sel(times=time.values).item()

        if PC == True: 
            if (45 <= lat <= 90) and ((90 <= lon <= 180) or (-180 <=lon <= -90)):
                timesList.append(time.values) 
        else: 
            if (45 <= lat <= 90) and (-90 <= lon <= 90):
                timesList.append(time.values)
    return timesList


#NORTH_ATLANTIC_JLa = Domain(north=80, south=67, east=60, west=-30)
#CENTRAL_ARCTIC_JLa = Domain(north=90, south=80, east=180, west=-180)
#BEAUFORT_SIBERIAN_JLa = Domain(north=80, south=70, east=-120, west=120)

def combine_times_circ(objs_):
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


    #Atlantic Hemisphere
    ATL_times = []
    for nr in range(len(objs_)):
        ATL_time=extract_hemisphere_times(objs_[nr])
        ATL_times.append(ATL_time)
    
    ATL_combined_times = np.concatenate(ATL_times)

    
    #Pacific Hemisphere
    PC_times = []
    for nr in range(len(objs_)):
        PC_time=extract_hemisphere_times(objs_[nr],True)
        PC_times.append(PC_time)
    
    PC_combined_times = np.concatenate(PC_times)

    return NA_combined_times, CA_combined_times, BS_combined_times,ATL_combined_times, PC_combined_times



def plot_circulation(objs_final,regimes,df,Arctic=False): 
    '''Function to visualize the circulation regime in each object 
    '''
    fig, axes = plt.subplots(12, 1, figsize=(6, 9))
    axes_flat = axes.flatten()
    plt.subplots_adjust(hspace=2.6)

    #Plot the sequence of regimes for each object
    for i, obj in enumerate(objs_final[0:12]):
        ax = axes_flat[i]
        
        #Object ID 
        obj_id = obj.id_.item()
    
        #Get dates (only once) in which object exists in 2 formats 
        obj_times = pd.to_datetime(obj.times.dt.strftime('%Y-%m-%d %H:%M:%S').values)
        obj_times_6h = obj_times[obj_times.hour.isin([0,6,12,18])]
        obj_dates_unique = np.unique(obj_times_6h.date)
        obj_times_12h = obj_times_6h[obj_times_6h.hour == 12]
    
        for j, date in enumerate(obj_times_6h):
            #Extract regime and color for the current date
            regime = df['cluster'][df['date'] == date].item()        
            regime_color = regimes[f'{regime}']['color']
    
            #Extract domain information
            domain = df.loc[df['date'] == date, 'domain'].values[0]
    
            #Plot regime and domain
            #Regime in color
            ax.bar(j, 1, width=0.85, color=regime_color, align='center', edgecolor='none')
            #Domain through stripes/pattern
            ax.bar(j, 1, width=0.85, color='none', align='center', 
                   edgecolor=domain_color[domain], hatch=domain_hatch[domain], linewidth=1.5)
            
        #Labels and title
        ax.set_title(f"ID: {obj_id}", fontsize=8,loc = 'left')
        ax.set_xticks(range(len(obj_times_6h)))
        ax.set_xticks([np.where(obj_times_6h == d)[0][0] for d in obj_times_12h])
        ax.set_xticklabels([d.strftime('%y/%m/%d') for d in obj_times_12h], fontsize=7)
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    
    # Add images as legend
    if Arctic:
        image_positions = [0.8, 0.65, 0.50, 0.35, 0.20,0.05]  # Vertical
        width = 0.2
        height = 0.2
    else: 
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
        ax_img.text(0.5, 1.05, info['name'], ha='center', va='bottom', fontsize=8, color='k', transform=ax_img.transAxes)

    #Add second legend to define domain patterns
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 45), ncol=3)
    
    #Add title
    fig.text(0.3,0.98,'Circulation Regimes for WAIs 1979-2022',fontsize=15)
    #plt.savefig('/work/aa0049/a271122/ERA5/CirculationRegime_clusters/atlantic_circulation_regimes.png', bbox_inches='tight', dpi=300)
    plt.show()



# Function to compute cluster percentages
def compute_cluster_counts(df, clusters):
    return [(len(df[df['cluster'] == c]) / len(df) * 100) for c in clusters]

def plot_cluster_distribution(df, title,bar_colors,Arctic= False):
    fig, axs = plt.subplots(2, 2, figsize=(7, 8))
    axs = axs.flatten()

    if Arctic:
        clusters =[1, 2, 3, 4, 5, 6]
    else: 
        clusters = [1, 2, 3, 4, 5]
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
        ax.set_ylim(0, 65)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Percentage [%]')
        ax.set_title(f"{domain}")
        ax.yaxis.grid(True, linestyle="--", alpha=0.7,zorder=1)
        
    # Add global title
    fig.suptitle(f'Distribution of {title} Circulation Clusters (1979-2022)', y=1,fontsize=14)
    
    plt.tight_layout()
    #plt.savefig('/work/aa0049/a271122/ERA5/CirculationRegime_clusters/distriburion_atlantic_clusters.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_cluster_distribution(df, title,bar_colors,Arctic= False):
    fig, axs = plt.subplots(2, 2, figsize=(7, 8))
    axs = axs.flatten()

    if Arctic:
        clusters =[1, 2, 3, 4, 5, 6]
    else: 
        clusters = [1, 2, 3, 4, 5]
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
        ax.set_ylim(0, 65)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Percentage [%]')
        ax.set_title(f"{domain}")
        ax.yaxis.grid(True, linestyle="--", alpha=0.7,zorder=1)
        
    # Add global title
    fig.suptitle(f'Distribution of {title} Circulation Clusters (1979-2022)', y=1,fontsize=14)
    
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


def get_normalized_frequency(objs_final,df_dates):
    # Define all cluster numbers from your regime definitions
    all_cluster_nums = [int(k) for k in atlantic_regimes.keys()]
    
    # Initialize total counters per domain and cluster with all clusters preset to 0.0
    cluster_weights = {
        'NA': defaultdict(float, {cl: 0.0 for cl in all_cluster_nums}),
        'CA': defaultdict(float, {cl: 0.0 for cl in all_cluster_nums}),
        'BS': defaultdict(float, {cl: 0.0 for cl in all_cluster_nums}),
    }
    
    # Process each object
    for obj in objs_final:
        obj_times = pd.to_datetime(obj.times.dt.strftime('%Y-%m-%d %H:%M:%S').values)
        obj_times_6h = obj_times[obj_times.hour.isin([0, 6, 12, 18])]
    
        # Track clusters per domain for this object
        domain_clusters = {'NA': [], 'CA': [], 'BS': []}
    
        for date in obj_times_6h:
            row = df_dates[df_dates['date'] == date]
            if not row.empty:
                domain = row['domain'].values[0]
                cluster = row['cluster'].values[0]
                if domain in domain_clusters:
                    domain_clusters[domain].append(cluster)
    
        # Normalize and add weights
        for domain, clusters in domain_clusters.items():
            if clusters:
                counts = pd.Series(clusters).value_counts(normalize=True)
                #print(f'{obj.id_.values}: {domain}: {counts}')
                for cluster_num in all_cluster_nums:
                    cluster_weights[domain][cluster_num] += counts.get(cluster_num, 0.0)
    
    # Create mapping from cluster number to cluster name
    cluster_num_to_name = (
        df_dates[['cluster', 'cluster_names']]
        .drop_duplicates()
        .set_index('cluster')['cluster_names']
        .to_dict()
    )
    
    # Convert weights to named version (ensuring all clusters are included)
    named_cluster_weights = {
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
    named_cluster_weights_perc = {}
    for domain, cluster_dict in cluster_weights.items():
        total_weight = sum(cluster_dict.values())
        normalized = {cluster_num_to_name[k]: (v / total_weight) * 100 for k, v in cluster_dict.items()}
    
        # Make sure all cluster names are present, including 0.0 entries
        for k in cluster_num_to_name:
            if cluster_num_to_name[k] not in normalized:
                normalized[cluster_num_to_name[k]] = 0.0
    
        named_cluster_weights_perc[domain] = normalized
    
    return named_cluster_weights_perc

def plot_hourly_hist(named_cluster_weights,counts_all,name,Pacific=False):
    if Pacific:
        bar_colors = ['rosybrown', 'firebrick', 'peru', 'teal', 'indigo']
        regime_names = ['PT', 'AH', 'AL', 'PWT', 'ALR']
        domain_names = {'NA':"North Atlantic",'CA': "Central Arctic", 'BS':"Beaufort/Siberian Sea"}

    else: 
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
        ax.set_ylim(0, 65)
        ax.set_ylabel('Normalized Frequency [%]')
        ax.set_title(f"{domain_names[domain]}")
        ax.set_xticks(clusters)
        ax.set_xticklabels(regime_names)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7,zorder=1)
    
        #Add Climatology clusters
        ax.bar(clusters, counts_all, color='grey', edgecolor='grey', width=0.5,align= 'edge',zorder=1,alpha=0.3)
    
        
    # Add global title
    #fig.suptitle(f'Distribution of Atlantic Circulation Clusters ERA5(1979-2022)', y=0.99,fontsize=14)
    
    '''    
    
    image_positions_h = [0, 0.29, 2*0.29, 3*0.29, 4*0.29]  # horizontal
    width = 0.3
    height = 0.3
    
    regimes = atlantic_regimes
    for i, (name, info) in enumerate(regimes.items()):
        img = mpimg.imread(info['image'])
        
        #positions
        ax_img = fig.add_axes([image_positions_h[i], 0, width, height])  #[left, bottom, width, height]
        ax_img.imshow(img)
        ax_img.axis('off')
        
        #Add frame and regime title
        rect = Rectangle((0, 0), 1.01, 1.01, linewidth=9, edgecolor=info['color'], facecolor='none', transform=ax_img.transAxes)
        ax_img.add_patch(rect)
        ax_img.text(0.5, 1.05, f"{info['name']}", ha='center', va='bottom', fontsize=12, color='k', transform=ax_img.transAxes)'''
    
    plt.savefig(f'/work/aa0049/a271122/ERA5/CirculationRegime_clusters/ERA5_distribution_{name}_clusters_new.png', bbox_inches='tight', dpi=300)
    plt.show()

# Function to compute cluster percentages
def compute_cluster_counts(df, clusters):
    return [(len(df[df['cluster'] == c]) / len(df) * 100) if len(df) > 0 else 0 for c in clusters]

def plot_cluster_distribution_withClusters(df, title,bar_colors,regimes,Arctic= False,Atlantic=False,Pacific=False):
    fig, axs = plt.subplots(2, 2, figsize=(8, 10))
    axs = axs.flatten()

    if Arctic:
        clusters =[1, 2, 3, 4, 5, 6]
    elif Atlantic: 
        clusters = [1, 2, 3, 4, 5]
        regime_names = ['NAO+', 'ScBl', 'AT', 'NAO-', 'DP']
    else:
        clusters =[1, 2, 3, 4, 5]
        regime_names = ['cl1', 'cl2', 'cl3', 'cl4', 'cl5']
        
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
        ax.set_ylim(0, 65)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Percentage [%]')
        ax.set_title(f"{domain}")
        ax.set_xticks(clusters)
        ax.set_xticklabels(regime_names)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7,zorder=1)
        
    # Add global title
    fig.suptitle(f'Distribution of {title} Circulation Clusters (1979-2022)', y=0.95,fontsize=14)

        
    # Add images as legend
    if Arctic:
        image_positions = [0.8, 0.65, 0.50, 0.35, 0.20,0.05]  # Vertical
        width = 0.2
        height = 0.2
    else: 
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
        ax_img.text(0.5, 1.05, f"{info['name']}", ha='center', va='bottom', fontsize=12, color='k', transform=ax_img.transAxes)

    #plt.savefig('/work/aa0049/a271122/ERA5/CirculationRegime_clusters/distriburion_atlantic_clusters.png', bbox_inches='tight', dpi=300)
    plt.show()
