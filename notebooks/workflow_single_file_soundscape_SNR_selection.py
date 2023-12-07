#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated 3 November 2022

Authors : Sylvain Haupert

"""

from IPython import get_ipython
# Clear all the variables
get_ipython().magic('reset -sf')

import shutil
import sys
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.close("all")

# package bambird
bambird_path   = Path('../../bamscape')
os.sys.path.append(bambird_path.as_posix())
import bambird
import bambird.config as cfg

# %%
# Define constants
# ----------------

# Temporary saved directory
TEMP_DIR        = Path('./temporary_data')
# name of the dataset
DATASET_NAME    = 'maad' # maad dB@jahra dB@risoux dB@petchora VLIZ ./
# Choose the name of the ROIs dataset
ROIS_NAME       = Path(str(DATASET_NAME) +'_ROIS')
# Name of the dataset. 
DATASET_DIR     = Path('/home/haupert/DATA/mes_projets_data/XPRIZE/dataset') / DATASET_NAME
# Name of the config file
CONFIG_FILE     = 'config_soundscape.yaml' # config_soundscape

# After the process, remove the audio that were saved during the process ?
CLEAN = True

# %%
if __name__ == '__main__':
    
    # Load the configuration file    
    params = cfg.load_config(CONFIG_FILE)

#%%    

    # Grab audio
    # -------------------
    df = bambird.grab_audio_to_df(path             =DATASET_DIR, 
                                  audio_format     ='wav',
                                  verbose          =True)
    
#%% 

    # Extract ROIS
    # -------------------------------
    
    # # select a random file
    df_single = df.sample(n=1).squeeze()
    # select a file by its filename
    # df_single = df[df['filename'] == 'RSB_20141214_022100_left.wav'] 
        
    # ROIS extraction of a single file
    df_rois_single = bambird.single_file_extract_rois (
                        audio_path  = df_single.squeeze().fullfilename,
                        fun         = params['PARAMS_EXTRACT']['FUNC'],
                        params      = params['PARAMS_EXTRACT'],
                        save_path   = TEMP_DIR / ROIS_NAME,
                        display     = False,
                        verbose     = True)
    
    # test if no ROI was found => break
    if len(df_rois_single) == 0 :
        print ("No ROI was found in the audio file {}".format(df_single.squeeze().fullfilename))
        sys.exit()
         
#%%
    # Compute features for each ROIS
    # -------------------------------

    ###### OPTIONAL
    # select a unique ROI
    path_to_unique_roi = df_rois_single.sample(n=1).fullfilename_ts.values[0]
    # Test the process on a single ROI file
    df_features_single_roi = bambird.compute_features(
                    audio_path          = path_to_unique_roi,
                    params              = params['PARAMS_FEATURES'],
                    display             = True,
                    verbose             = True
                    )
    ###### END OPTIONAL
    
    # Test the process the ROIs of a single file
    df_features_single, csv_features = bambird.multicpu_compute_features(
                    dataset     = df_rois_single,
                    params      = params['PARAMS_FEATURES'],
                    save_path   = TEMP_DIR / ROIS_NAME,
                    overwrite   = True,
                    verbose     = True)
        
#%%        
    #  Cluster ROIS
    # -------------------------------

    #!!!!!!!!!!!!!!
    # Select the SNR for the clustering
    MIN_SNR = 10
    
    df_cluster, csv_cluster = bambird.find_cluster(
                    dataset = df_features_single[df_features_single.snr>MIN_SNR],
                    params  = params['PARAMS_CLUSTER'],
                    display = False,
                    verbose = True
                    )
       
#%%    
    # Display the ROIS
    # -------------------------------
    
    bambird.overlay_rois(
                        cluster         = df_cluster,
                        params          = params['PARAMS_EXTRACT'],
                        column_labels   = 'cluster_number', #auto_label
                        unique_labels   = np.sort(df_cluster.cluster_number.unique()),
                        # filename        = 'EBA_20141213_222100_right.wav', # None df_single.filename
                        random_seed     = None,
                        verbose         = True
                        )

#%%    
    # Remove files
    # -------------------------------

    if CLEAN :
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
            