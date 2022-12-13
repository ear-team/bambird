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
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.close("all")

# import the package bambird
import bambird
import bambird.config as cfg

# %%
# Define constants
# ----------------

# Temporary saved directory
TEMP_DIR        = "./build_dataset_temp"
DATASET_NAME    = Path('WORKFLOW_SINGLE_FILE')
ROIS_NAME       = Path(str(DATASET_NAME) +'_ROIS')

# List of species to build a clean dataset
SCIENTIC_NAME_LIST = [
                        "Columba palumbus",
                        # "Regulus regulus",
                        # "Phylloscopus collybita",
                        # "Anthus triviali", 
                        # "Fringilla coelebs", 
                        # "Troglodytes troglodytes", 
                        # "Phoenicurus phoenicurus", 
                        # "Strix aluco", 
                        # "Aegithalos caudatus",
                      ]

CONFIG_FILE = '../config_default.yaml' 

# After the process, remove the audio that were saved during the process ?
CLEAN = True

# %%
if __name__ == '__main__':

    # Load the configuration file    
    params = cfg.load_config(CONFIG_FILE)

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
    
    # # select a random file
    df_single = df_xc.sample(n=1).squeeze()
    # select a file by its XC name
    # df_single = df_xc[df_xc['filename'] == 'XC574126.mp3'].squeeze() 
        
    # ROIS extraction of a single file
    df_rois_single = bambird.single_file_extract_rois (
                        audio_path  = df_single.fullfilename,
                        fun         = params['PARAMS_EXTRACT']['FUNC'],
                        params      = params['PARAMS_EXTRACT'],
                        save_path   = TEMP_DIR / ROIS_NAME,
                        display     = True,
                        verbose     = True)
    
    # ROIS extraction of the full dataset
    df_rois, csv_rois = bambird.multicpu_extract_rois(
                        dataset     = df_xc,
                        params      = params['PARAMS_EXTRACT'],
                        save_path   = TEMP_DIR / ROIS_NAME,
                        overwrite   = True,
                        verbose     = True
                        )
    
#%%
    # Compute features for each ROIS
    # -------------------------------
    
    # Test if at least 1 ROI was found
    if len(df_rois_single) > 0 :
        # select a unique ROI
        path_to_unique_roi = df_rois_single.sample(n=1).fullfilename_ts.values[0]
        # Test the process on a single ROI file
        df_features_single_roi = bambird.compute_features(
                        audio_path  = path_to_unique_roi,
                        params      = params['PARAMS_FEATURES'],
                        display     = True,
                        verbose     = True
                        )
        
        # Test the process the ROIs of a single file
        df_features_single, csv_features = bambird.multicpu_compute_features(
                        dataset     = df_rois_single,
                        params      = params['PARAMS_FEATURES'],
                        save_path   = TEMP_DIR / ROIS_NAME,
                        overwrite   = True,
                        verbose     = True)
        
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
    
    dataset = df_features # df_features df_features_single
    
    try : 
        df_cluster, csv_clusters = bambird.find_cluster(
                        dataset = dataset,
                        params  = params['PARAMS_CLUSTER'],
                        display = True,
                        verbose = True
                        )
    except:
       df_cluster = dataset 
       df_cluster['auto_label'] = 0
       df_cluster['cluster_number'] = -1
       
#%%    
    # Display the ROIS
    # -------------------------------
    
    bambird.overlay_rois(
                        cluster         = df_cluster,
                        params          = params['PARAMS_EXTRACT'],
                        column_labels   = 'cluster_number', #auto_label
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
            