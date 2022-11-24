#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated 3 November 2022

Authors : Felix Michaud and Sylvain Haupert

"""

from IPython import get_ipython
print(__doc__)
# Clear all the variables
get_ipython().magic('reset -sf')

import shutil
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.close("all")

# import the package bambird
import bambird

# %%
# Define constants
# ----------------

# Temporary saved directory
TEMP_DIR        = "./build_dataset_temp"
DATASET_NAME    = Path('WORKFLOW_MULTIPLE_SPECIES')
ROIS_NAME       = Path(str(DATASET_NAME) +'_ROIS')

# List of species to build a clean dataset
SCIENTIC_NAME_LIST = [
                        "Regulus regulus",
                        "Phylloscopus collybita",
                        "Anthus triviali", 
                        "Fringilla coelebs", 
                        "Troglodytes troglodytes", 
                        "Phoenicurus phoenicurus", 
                        "Strix aluco", 
                        "Aegithalos caudatus",
                      ]

CONFIG_FILE = '../config_default.yaml' 

# After the process, remove the audio that were saved during the process ?
CLEAN = False

# %%
if __name__ == '__main__':

    with open(CONFIG_FILE) as f:
        params = yaml.load(f, Loader=bambird.get_loader())

#%%    
    # Query Xeno-Canto
    # ----------------
    df_dataset = bambird.query_xc(
                        species_list    = SCIENTIC_NAME_LIST,
                        params          = params['PARAMS_XC'],
                        random_seed     = params['RANDOM_SEED'],
                        verbose         = True
                        )
    
    # Download audio Xeno-Canto
    # -------------------------
    df_xc, csv_xc  = bambird.download_xc (
                        df_dataset      = df_dataset,
                        rootdir         = TEMP_DIR, 
                        dataset_name    = DATASET_NAME, 
                        csv_filename    = params['PARAMS_XC']['CSV_XC_FILE'],
                        overwrite       = True,
                        verbose         = True
                        )
    
#%% 
    
    # Extract ROIS
    # -------------------------------
    
    # ROIS extraction of the full dataset
    df_rois, csv_rois = bambird.multicpu_extract_rois(
                        dataset     = df_xc,
                        fun         = params['PARAMS_EXTRACT']['FUNC'],
                        params      = params['PARAMS_EXTRACT'],
                        save_path   = TEMP_DIR / ROIS_NAME,
                        overwrite   = True,
                        verbose     = True
                        )
    
#%%
    # Compute features for each ROIS
    # -------------------------------
        
    # Test if at least 1 ROI was found     
    if len(df_rois) > 0 :    
        # compute the features on the full dataset       
        df_features, csv_features = bambird.multicpu_compute_features(
                        dataset     = df_rois,
                        params      = params['PARAMS_FEATURES'],
                        save_path   = TEMP_DIR / ROIS_NAME,
                        overwrite   = True,
                        verbose     = True)
        
#%%        
    #  Cluster ROIS
    # -------------------------------
    
    # with dataframe or csv file
    
    dataset = df_features 
    
    try : 
        df_cluster = bambird.find_cluster(
                        dataset = dataset,
                        params  = params['PARAMS_CLUSTER'],
                        display = True,
                        verbose = True
                        )
    except:
       df_cluster = df_features 
       df_cluster['auto_label'] = 0
       df_cluster['cluster_number'] = -1
       
#%%    
    # Display the ROIS
    # -------------------------------
    
    bambird.overlay_rois(
                        cluster         = df_cluster,
                        params          = params['PARAMS_EXTRACT'],
                        column_labels   = 'cluster_number', #auto_label cluster_number
                        unique_labels   = np.sort(df_cluster.cluster_number.unique()),
                        filename        = None,
                        random_seed     = None,
                        verbose         = True
                        )

#%%    
    # Remove files
    # -------------------------------
    
    if CLEAN :
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
            