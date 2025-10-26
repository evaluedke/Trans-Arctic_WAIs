###### functions used in model comparison, lifecycle plots throughout the century
# author: eva luedke
# 2025

################IMPORT PACKAGES########################

import xarray as xr
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import re

##############DEFINITIONS#############################

ERA5_clim_path='/work/aa0049/a271122/ERA5/'
ICON_ERA5_clim_path = '/work/aa0049/a271122/ICON_ERA5/'
ICON_CNRM_clim_path = '/work/aa0049/a271122/ICON_CNRM/'
ICON_NorESM_clim_path = '/work/aa0049/a271122/ICON_NorESM/'

model = ['ERA5','ICON_ERA5','ICON_CNRM','ICON_NorESM']

#######################################################
##################FUNCTIONS############################
#######################################################

def load_data(var,var2):
    '''Function to load anomalies in historical period for all 4 datasets'''

    #Open shape_lists
    with open(f'{ERA5_clim_path}shapelists/{var}_anomaly_list.pkl', 'rb') as f:
       ERA5_anomaly_list = pickle.load(f) 
    with open(f'{ICON_ERA5_clim_path}shapelists/{var2}_anomaly_list.pkl', 'rb') as f:
       ICON_ERA5_anomaly_list = pickle.load(f)
    with open(f'{ICON_CNRM_clim_path}shapelists/{var}_anomaly_list.pkl', 'rb') as f:
       ICON_CNRM_anomaly_list = pickle.load(f)
    with open(f'{ICON_NorESM_clim_path}shapelists/{var}_anomaly_list.pkl', 'rb') as f:
       ICON_NorESM_anomaly_list = pickle.load(f)

    #Concatenate lists into 3 groups (one per domain)
    ERA5_anomalies = [np.concatenate([da.values.flatten() for da in group]) for group in ERA5_anomaly_list]
    ICON_ERA5_anomalies = [np.concatenate([da.values.flatten() for da in group]) for group in ICON_ERA5_anomaly_list]
    ICON_CNRM_anomalies = [np.concatenate([da.values.flatten() for da in group]) for group in ICON_CNRM_anomaly_list]
    ICON_NorESM_anomalies = [np.concatenate([da.values.flatten() for da in group]) for group in ICON_NorESM_anomaly_list]

    return ERA5_anomalies, ICON_ERA5_anomalies, ICON_CNRM_anomalies, ICON_NorESM_anomalies


def plot_violin(ERA5_anomalies, ICON_ERA5_anomalies, ICON_CNRM_anomalies, ICON_NorESM_anomalies,maintitle,ylabel,ymin,ymax):
    '''Function to plot model data as violin plots for different models'''
    titles = ['North Atlantic', 'Central Arctic', 'Beaufort Sea']

    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
      
    
    for idx, (title,ax) in enumerate(zip(titles,axs)):
        vp1 = ax.violinplot(ERA5_anomalies[idx],positions=[1], showmeans=True, showmedians=False)
        vp2 = ax.violinplot(ICON_ERA5_anomalies[idx],positions=[2], showmeans=True, showmedians=False)
        vp3 = ax.violinplot(ICON_CNRM_anomalies[idx], positions=[3], showmeans=True, showmedians=False)
        vp4 = ax.violinplot(ICON_NorESM_anomalies[idx], positions=[4], showmeans=True, showmedians=False)
    
        
        # Manually set colors for each dataset
        #colors = ['#92C5DE', '#A6D854', '#F4A582', '#DDA0DD']
        #colors = ['#377EB8', '#FF7F00', '#4DAF4A', '#984EA3']
        colors = ["#1B9E77","#D95F02", "#7570B3","#E6AB02"]
        #colors = ["#377EB8","#E41A1C", "#4DAF4A","#984EA3"]
        #colors = ["#6A3D9A","#F46D43","#66A61E","#E6AB02"]
        #colors = ['#1B9E77', '#D95F02', '#7570B3', '#E6AB02']
        
        for vp, color in zip([vp1, vp2, vp3, vp4], colors):
            for partname in ('cbars', 'cmins', 'cmaxes', 'bodies'):
                parts = vp[partname]
                if isinstance(parts, list):  # For 'bodies', it's a list
                    for pc in parts:
                        pc.set_facecolor(color)
                        #pc.set_edgecolor('black')
                        pc.set_alpha(0.6)
                else:
                    parts.set_color(color)
            vp['cmeans'].set_color('k') #'k'
            
            # Customize plot
            ax.set_xticks([1, 2, 3,4])
            ax.set_xticklabels(["ERA5","ICON_ERA5", "ICON_CNRM", "ICON_NorESM"],rotation=90)
            ax.set_ylim([ymin,ymax])
            ax.set_xlabel(title)
            #ax.grid(color='grey', linestyle='--', linewidth=0.8,alpha=0.6)#,zorder=10)
          
            if idx == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.yaxis.set_visible(False)  
    
    # Create a legend manually
    legend_labels = ["ERA5","ICON_ERA5", "ICON_CNRM", "ICON_NorESM"]
    legend_patches = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    ax.legend(legend_patches, legend_labels, loc="upper right",bbox_to_anchor=(1.7, 1),fontsize=9)
    
    
    plt.subplots_adjust(wspace=0.05)
    fig.suptitle(maintitle, fontweight='bold')#,y=0.99)
    
    plt.show()


def plot_box(ERA5_anomalies, ICON_ERA5_anomalies, ICON_CNRM_anomalies, ICON_NorESM_anomalies,maintitle,ylabel,ymin,ymax): 
    '''Function to plot model data as box plots for model comparison'''
    titles = ['North Atlantic', 'Central Arctic', 'Beaufort Sea']

    fig, axs = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
    
    # Define colors for each dataset
    #colors = ['#377EB8', '#FF7F00', '#4DAF4A', '#984EA3']
    colors = ["#1B9E77","#D95F02", "#7570B3","#E6AB02"]
    
    for idx, (title, ax) in enumerate(zip(titles, axs)):
        # Create boxplots
        bp = ax.boxplot(
            [ERA5_anomalies[idx], ICON_ERA5_anomalies[idx], ICON_CNRM_anomalies[idx], ICON_NorESM_anomalies[idx]],
            positions=[1, 2, 3, 4], 
            patch_artist=True,  # Allows setting box colors
            showmeans=False,
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
    
        # Set axis labels and formatting
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["ERA5", "ICON_ERA5", "ICON_CNRM", "ICON_NorESM"], rotation=90)
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel(title)
    
        if idx == 0:
            ax.set_ylabel(ylabel)
    
    # Create a legend manually
    legend_labels = ["ERA5", "ICON_ERA5", "ICON_CNRM", "ICON_NorESM"]
    legend_patches = [plt.Line2D([0], [0], color=color, marker='s', markersize=10, linestyle='None') for color in colors]
    
    fig.legend(legend_patches, legend_labels, loc="upper right", bbox_to_anchor=(1.05, 0.89), fontsize=9)
    
    plt.subplots_adjust(wspace=0.05)
    fig.suptitle(maintitle, fontweight='bold')
    
    plt.show()


def plot_box_years(anomalies_84_14,anomalies_15_39,anomalies_40_69,anomalies_70_99,maintitle,ylabel,ymin,ymax): 
    '''Function to plot development over centuries in one model as box plot'''
    
    titles = ['North Atlantic', 'Central Arctic', 'Beaufort Sea']

    fig, axs = plt.subplots(1, 3, figsize=(8, 10), sharey=True)
    
    # Define colors for each dataset
    #colors = ['#377EB8', '#FF7F00', '#4DAF4A', '#984EA3']
    colors = ["#1B9E77","#D95F02", "#7570B3","#E6AB02"]
    colors = ["darkblue","darkred", "#D95F02"]
    
    for idx, (title, ax) in enumerate(zip(titles, axs)):
        # Create boxplots
        bp = ax.boxplot(
            [anomalies_84_14[idx],anomalies_15_39[idx],anomalies_40_69[idx],anomalies_70_99[idx]],
            positions=[1, 2, 3, 4], 
            patch_artist=True,  # Allows setting box colors
            showmeans=False
        )
        
            
        # Set colors for boxes, whiskers, caps, medians, and outliers
        for i, (box, color) in enumerate(zip(bp['boxes'], colors)):
            box.set(facecolor=color, alpha=1, edgecolor='black')
    
            # Whiskers and caps
            for whisker in bp['whiskers'][2 * i:2 * i + 2]:
                whisker.set(color=color, linewidth=1.2)
            for cap in bp['caps'][2 * i:2 * i + 2]:
                cap.set(color=color, linewidth=1.2)
    
            # Medians
            bp['medians'][i].set(color='black', linewidth=1.5)
    
            # Outlier points
            bp['fliers'][i].set(marker='o', markerfacecolor=color, markeredgecolor=color, alpha=1, markersize=5)
    
        # Set axis labels and formatting
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["1985-2014", "2015-2039", "2040-2069", "2070-2099"], rotation=90)
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel(title)
    
        if idx == 0:
            ax.set_ylabel(ylabel)
    
    # Create a legend manually
    legend_labels = ["1985-2014", "2015-2039", "2040-2069", "2070-2099"]
    legend_patches = [plt.Line2D([0], [0], color=color, marker='s', markersize=10, linestyle='None') for color in colors]
    
    fig.legend(legend_patches, legend_labels, loc="upper right", bbox_to_anchor=(1.05, 0.89), fontsize=9)
    
    plt.subplots_adjust(wspace=0.05)
    fig.suptitle(maintitle, fontweight='bold')
    
    plt.show()


def plot_box_years_new(anomalies_84_14,anomalies_15_39,anomalies_40_69,anomalies_70_99,
                       clim_list_84_14,clim_list_15_39,clim_list_40_69,clim_list_70_99,
                       var,maintitle,ylabel,ymin,ymax,ICON_clim_path,clim=False): 
    '''New function to plot development over the centuries in one model as box plot with adjusted colors, mean and median and outliers 
    defined as outside of 98% of the points'''
    
    plt.rcParams.update({
    'axes.titlesize': 24,        # Title size
    'axes.labelsize': 21,         # Axis label size
    'lines.linewidth': 3,         # Line width
    'lines.markersize': 10,       # Marker size for lines
    'xtick.labelsize': 19,        # X-tick label size
    'ytick.labelsize': 16         # Y-tick label size
    })
    
    
    titles = ['North Atlantic', 'Central Arctic', 'Beaufort Sea']

    fig, axs = plt.subplots(1, 3, figsize=(8, 10), sharey=True)
    
    # Define colors for each dataset
    colors = ["darkblue","darkred", "#D95F02"]
    alpha_ = [1,0.8,0.6,0.4]
    
    for idx, (title, ax) in enumerate(zip(titles, axs)):
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

        if clim:
            positions = [1,2,3,4]
            
            if var=='pr':
                clim_84 = clim_list_84_14[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values*3600
                clim_15 = clim_list_15_39[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values*3600
                clim_40 = clim_list_40_69[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values*3600
                clim_70 = clim_list_70_99[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values*3600
                
                ax.hlines([clim_84,
                            clim_15,
                            clim_40,
                            clim_70],
                           [p - 0.3 for p in positions],  # xmin per box
                           [p + 0.3 for p in positions],  # xmax per box
                            colors='black', linewidth=2,linestyle='--')
            else:    
                ax.hlines([clim_list_84_14[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values,
                        clim_list_15_39[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values,
                        clim_list_40_69[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values,
                        clim_list_70_99[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values],
                       [p - 0.3 for p in positions],  # xmin per box
                       [p + 0.3 for p in positions],  # xmax per box
                        colors='black', linewidth=2,linestyle='--')

    
        
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

    
        if idx == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.yaxis.set_visible(False)  
    
    plt.subplots_adjust(wspace=0.05)
    fig.suptitle(maintitle, fontweight='bold',y=0.92,fontsize=20)


    safe_title = re.sub(r'[^\w\-.]', '_', maintitle)  # Replace unsafe characters
    plt.savefig(f'{ICON_clim_path}/{safe_title}.png',dpi=300,bbox_inches='tight')
    
    plt.show()

def plot_box_and_clim(anomalies_84_14,anomalies_15_39,anomalies_40_69,anomalies_70_99,
                      clim_list_84_14,clim_list_15_39,clim_list_40_69,clim_list_70_99,
                      var,maintitle,ylabel,ymin,ymax,ICON_clim_path,clim_min,clim_max): 
    '''Function to plot development over the centuries in one model as box plot with adjusted colors, mean and median and outliers 
    defined as outside of 98% of the points as well as climatology in a second axis'''


    plt.rcParams.update({
    'axes.titlesize': 24,        # Title size
    'axes.labelsize': 21,         # Axis label size
    'lines.linewidth': 3,         # Line width
    'lines.markersize': 10,       # Marker size for lines
    'xtick.labelsize': 19,        # X-tick label size
    'ytick.labelsize': 16         # Y-tick label size
    })
    
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

            print(f'Climatologies {title}:  84: {clim_84:.2f} \n 15: {clim_15:.2f} \n 40: {clim_40:.2f}\n 70: {clim_70:.2f}')
            ax2.hlines([clim_84,
                        clim_15,
                        clim_40,
                        clim_70],
                       [p - 0.3 for p in positions],  # xmin per box
                       [p + 0.3 for p in positions],  # xmax per box
                        colors='black', linewidth=2,linestyle='--')
        else:
            clim_84 = clim_list_84_14[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values
            clim_15 = clim_list_15_39[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values
            clim_40 = clim_list_40_69[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values
            clim_70 = clim_list_70_99[idx][var].mean(dim='time').mean(dim=('rlat','rlon')).values

            print(f'Climatologies {title}: 84: {clim_84:.2f} \n 15: {clim_15:.2f} \n 40: {clim_40:.2f} \n 70: {clim_70:.2f}')
            
            ax2.hlines([clim_84,
                        clim_15,
                        clim_40,
                        clim_70],
                       [p - 0.3 for p in positions],  # xmin per box
                       [p + 0.3 for p in positions],  # xmax per box
                        colors='black', linewidth=2,linestyle='--')
            
            
        print(f'Anomalies {title}: 84: {anomalies_84_14[idx].mean():.2f} \n 15: {anomalies_15_39[idx].mean():.2f} \n 40: {anomalies_40_69[idx].mean():.2f} \n 70: {anomalies_70_99[idx].mean():.2f}')
        
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
        #ax.set_xticklabels(["", "", "", ""], rotation=90)

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
    fig.suptitle(maintitle, fontweight='bold',y=0.92,fontsize=20)


    safe_title = re.sub(r'[^\w\-.]', '_', maintitle)  # Replace unsafe characters
    #plt.tight_layout()
    plt.savefig(f'{ICON_clim_path}/{safe_title}_clim.png',dpi=300,bbox_inches='tight')
    
    plt.show()



def plot_violin_years(anomalies_84_14,anomalies_15_39,anomalies_40_69,anomalies_70_99,maintitle,ylabel,ymin,ymax):
    '''Plots development over centuries for one model as violin plot'''
    titles = ['North Atlantic', 'Central Arctic', 'Beaufort Sea']

    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
      
    for idx, (title,ax) in enumerate(zip(titles,axs)):
        vp1 = ax.violinplot(anomalies_84_14[idx],positions=[1], showmeans=True, showmedians=False)
        vp2 = ax.violinplot(anomalies_15_39[idx],positions=[2], showmeans=True, showmedians=False)
        vp3 = ax.violinplot(anomalies_40_69[idx], positions=[3], showmeans=True, showmedians=False)
        vp4 = ax.violinplot(anomalies_70_99[idx], positions=[4], showmeans=True, showmedians=False)
    
        
        # Manually set colors for each dataset
        #colors = ['#92C5DE', '#A6D854', '#F4A582', '#DDA0DD']
        #colors = ['#377EB8', '#FF7F00', '#4DAF4A', '#984EA3']
        colors = ["#1B9E77","#D95F02", "#7570B3","#E6AB02"]
        #colors = ["#377EB8","#E41A1C", "#4DAF4A","#984EA3"]
        #colors = ["#6A3D9A","#F46D43","#66A61E","#E6AB02"]
        #colors = ['#1B9E77', '#D95F02', '#7570B3', '#E6AB02']
        
        for vp, color in zip([vp1, vp2, vp3, vp4], colors):
            for partname in ('cbars', 'cmins', 'cmaxes', 'bodies'):
                parts = vp[partname]
                if isinstance(parts, list):  # For 'bodies', it's a list
                    for pc in parts:
                        pc.set_facecolor(color)
                        #pc.set_edgecolor('black')
                        pc.set_alpha(0.6)
                else:
                    parts.set_color(color)
            vp['cmeans'].set_color('k') #'k'
            
            # Customize plot
            ax.set_xticks([1, 2, 3,4])
            ax.set_xticklabels(["1985-2014", "2015-2039", "2040-2069", "2070-2099"],rotation=90)
            ax.set_ylim([ymin,ymax])
            ax.set_xlabel(title)
            #ax.grid(color='grey', linestyle='--', linewidth=0.8,alpha=0.6)#,zorder=10)
          
            if idx == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.yaxis.set_visible(False)  
    
    # Create a legend manually
    legend_labels = ["1985-2014", "2015-2039", "2040-2069", "2070-2099"]
    legend_patches = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    ax.legend(legend_patches, legend_labels, loc="upper right",bbox_to_anchor=(1.7, 1),fontsize=9)
    
    
    plt.subplots_adjust(wspace=0.05)
    fig.suptitle(maintitle, fontweight='bold')#,y=0.99)
    
    plt.show()

def Central_Arctic_plot_violin_years(anomalies_84_14,anomalies_15_39,anomalies_40_69,anomalies_70_99,maintitle,ylabel,ymin,ymax):
    '''Function to plot development over centuries only in the Central Arctic as violin plots'''
    titles = ['North Atlantic', 'Central Arctic', 'Beaufort Sea']

    fig, axs = plt.subplots(1, 1, figsize=(4, 6))
      
   
    vp1 = ax.violinplot(anomalies_84_14[1],positions=[1], showmeans=True, showmedians=False)
    vp2 = ax.violinplot(anomalies_15_39[2],positions=[2], showmeans=True, showmedians=False)
    vp3 = ax.violinplot(anomalies_40_69[3], positions=[3], showmeans=True, showmedians=False)
    vp4 = ax.violinplot(anomalies_70_99[4], positions=[4], showmeans=True, showmedians=False)

    
    # Manually set colors for each dataset
    #colors = ['#92C5DE', '#A6D854', '#F4A582', '#DDA0DD']
    #colors = ['#377EB8', '#FF7F00', '#4DAF4A', '#984EA3']
    colors = ["#1B9E77","#D95F02", "#7570B3","#E6AB02"]
    #colors = ["#377EB8","#E41A1C", "#4DAF4A","#984EA3"]
    #colors = ["#6A3D9A","#F46D43","#66A61E","#E6AB02"]
    #colors = ['#1B9E77', '#D95F02', '#7570B3', '#E6AB02']
    
    for vp, color in zip([vp1, vp2, vp3, vp4], colors):
        for partname in ('cbars', 'cmins', 'cmaxes', 'bodies'):
            parts = vp[partname]
            if isinstance(parts, list):  # For 'bodies', it's a list
                for pc in parts:
                    pc.set_facecolor(color)
                    #pc.set_edgecolor('black')
                    pc.set_alpha(0.6)
            else:
                parts.set_color(color)
        vp['cmeans'].set_color('k') #'k'
        
        # Customize plot
        ax.set_xticks([1, 2, 3,4])
        ax.set_xticklabels(["1985-2014", "2015-2039", "2040-2069", "2070-2099"],rotation=90)
        ax.set_ylim([ymin,ymax])
        ax.set_xlabel('Central Arctic')
        #ax.grid(color='grey', linestyle='--', linewidth=0.8,alpha=0.6)#,zorder=10)
      
        if idx == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.yaxis.set_visible(False)  
    
    # Create a legend manually
    legend_labels = ["1985-2014", "2015-2039", "2040-2069", "2070-2099"]
    legend_patches = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    ax.legend(legend_patches, legend_labels, loc="upper right",bbox_to_anchor=(1.7, 1),fontsize=9)
    
    
    plt.subplots_adjust(wspace=0.05)
    fig.suptitle(maintitle, fontweight='bold')#,y=0.99)
    
    plt.show()


def load_shapes(var,ICON_clim_path):
    '''Function to load previously created shapelists that can then be used for violin or box plots'''
    #Open shape_lists
    with open(f'{ICON_clim_path}shapelists/{var}_shape_list.pkl', 'rb') as f:
       shape_list_84_14 = pickle.load(f) 
    with open(f'{ICON_clim_path}ssp/shapelists/{var}_shape_list_15-39.pkl', 'rb') as f:
       shape_list_15_39 = pickle.load(f)
    with open(f'{ICON_clim_path}ssp/shapelists/{var}_shape_list_40-69.pkl', 'rb') as f:
       shape_list_40_69 = pickle.load(f)
    with open(f'{ICON_clim_path}ssp/shapelists/{var}_shape_list_70-99.pkl', 'rb') as f:
       shape_list_70_99 = pickle.load(f)
    
    
    #Concatenate lists into 3 groups (one per domain)
    shape_list_84_14 = [np.concatenate([da.values.flatten() for da in group]) for group in shape_list_84_14]
    shape_list_15_39 = [np.concatenate([da.values.flatten() for da in group]) for group in shape_list_15_39]
    shape_list_40_69 = [np.concatenate([da.values.flatten() for da in group]) for group in shape_list_40_69]
    shape_list_70_99 = [np.concatenate([da.values.flatten() for da in group]) for group in shape_list_70_99]
    
    return shape_list_84_14,shape_list_15_39,shape_list_40_69,shape_list_70_99


def load_anomalies(var,ICON_clim_path):
    '''Function to load previously computed anomaly lists for plotting development over centuries'''
    #Open shape_lists
    with open(f'{ICON_clim_path}shapelists/{var}_anomaly_list.pkl', 'rb') as f:
       anomaly_list_84_14 = pickle.load(f) 
    with open(f'{ICON_clim_path}ssp/shapelists/{var}_anomaly_list_15-39.pkl', 'rb') as f:
       anomaly_list_15_39 = pickle.load(f)
    with open(f'{ICON_clim_path}ssp/shapelists/{var}_anomaly_list_40-69.pkl', 'rb') as f:
       anomaly_list_40_69 = pickle.load(f)
    with open(f'{ICON_clim_path}ssp/shapelists/{var}_anomaly_list_70-99.pkl', 'rb') as f:
       anomaly_list_70_99 = pickle.load(f)

    #Concatenate lists into 3 groups (one per domain)
    anomaly_list_84_14 = [np.concatenate([da.values.flatten() for da in group]) for group in anomaly_list_84_14]
    anomaly_list_15_39 = [np.concatenate([da.values.flatten() for da in group]) for group in anomaly_list_15_39]
    anomaly_list_40_69 = [np.concatenate([da.values.flatten() for da in group]) for group in  anomaly_list_40_69]
    anomaly_list_70_99 = [np.concatenate([da.values.flatten() for da in group]) for group in anomaly_list_70_99]

    return anomaly_list_84_14,anomaly_list_15_39,anomaly_list_40_69,anomaly_list_70_99

def load_climatologies(var,ICON_clim_path):
    '''Function to load previously computed anomaly lists for plotting development over centuries'''
    #Open shape_lists
    with open(f'{ICON_clim_path}shapelists/{var}_clim_list.pkl', 'rb') as f:
       clim_list_84_14 = pickle.load(f) 
    with open(f'{ICON_clim_path}ssp/shapelists/{var}_clim_list_15-39.pkl', 'rb') as f:
       clim_list_15_39 = pickle.load(f)
    with open(f'{ICON_clim_path}ssp/shapelists/{var}_clim_list_40-69.pkl', 'rb') as f:
       clim_list_40_69 = pickle.load(f)
    with open(f'{ICON_clim_path}ssp/shapelists/{var}_clim_list_70-99.pkl', 'rb') as f:
       clim_list_70_99 = pickle.load(f)

    return clim_list_84_14,clim_list_15_39,clim_list_40_69,clim_list_70_99



