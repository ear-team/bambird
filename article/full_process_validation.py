#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated 19 October 2022

Authors : Felix Michaud and Sylvain Haupert

"""

from IPython import get_ipython
print(__doc__)
# Clear all the variables
get_ipython().magic('reset -sf')

import yaml
from pathlib import Path
import matplotlib.pyplot as plt
plt.close("all")
import pandas as pd

import bambird

# %%
# Define constants
# ----------------

RANDOM_SEED = 1979

# Choose the path to store the mp3
DIR_DATA        = Path('../../temporary_data')
# Choose the name of the dataset
DATASET_NAME    = Path('DATASET_PALEARTIC_PART1')
# Choose the name of the ROIs dataset
ROIS_NAME       = Path(str(DATASET_NAME) +'_ROIS_VALIDATION')
# Select the csv file with the metadata collected from xeno-canto (=> links to the mp3 to download)
XC_CSV_FILE     = Path('./data/validation') / 'xc_metadata.csv'
# Select the annotation file corresponding to the ROIs dataset
ANNOT_CSV_FILE  = Path('./data/validation') / 'manual_annotations.csv'
# Select the configuration file to segment, compute features and get the clusters
CONFIG_FILE     = 'config_article.yaml' 

# %%
if __name__ == '__main__':

    with open(CONFIG_FILE) as f:
        params = yaml.load(f, Loader=bambird.get_loader())
        
    # load the inital dataset with the metadata stored from XC
    df_dataset = pd.read_csv(XC_CSV_FILE, sep=';')
                
#%%

    # Download audio Xeno-Canto
    # -------------------------
    
    df_xc, csv_xc  = bambird.download_xc (
                    df_dataset    = df_dataset,
                    rootdir       = DIR_DATA,  
                    dataset_name  = DATASET_NAME,
                    overwrite     = True,
                    verbose       = True
                    )
 
#%%        
    
    # Extract ROIS
    # -------------------------------    
    
    # extract ROIS
    df_rois, csv_rois = bambird.multicpu_extract_rois(
                    dataset             =df_xc,
                    fun                 =params['PARAMS_EXTRACT']['FUNC'],
                    params              =params['PARAMS_EXTRACT'],
                    save_path           =DIR_DATA / ROIS_NAME,
                    overwrite           =True,
                    verbose             =True
                    )
                
#%%           
 
    # process all the ROIS
    # ---------------------------------------------------------------------

    # compute features        
    df_features, csv_features = bambird.multicpu_compute_features(
                    dataset             =df_rois,
                    params              =params['PARAMS_FEATURES'],
                    save_path           =DIR_DATA / ROIS_NAME,
                    overwrite           =True,
                    verbose             =True
                    )
        
#%%
    # Cluster ROIS
    # ---------------------------------------------------------------------

    # with dataframe or csv file
    df_cluster, csv_cluster = bambird.find_cluster(
                    dataset     =df_features,
                    params      =params['PARAMS_CLUSTER'],
                    save_path   =DIR_DATA / ROIS_NAME,
                    display     =False,
                    verbose     =True
                    )

    # Evaluation of the clustering : precision and recall
    # ---------------------------------------------------------------------
    
    df_scores, p, r, f, markers = bambird.cluster_eval(
                    df_cluster, 
                    path_to_csv_with_gt     = ANNOT_CSV_FILE,
                    colname_label_gt        ='manual_label',
                    verbose                 =True
                    )
        
#%%   
    # Display the ROIS
    # ---------------------------------------------------------------------
    filename = bambird.overlay_rois(
                    cluster     =df_cluster,
                    params      =params['PARAMS_EXTRACT'],
                    filename    =None,
                    random_seed =None,
                    verbose     =True
                    )
    
    # Display the ROIS with TP=1 TN = 2 FN = 3 FP = 4
    # ---------------------------------------------------------------------
    
    bambird.overlay_rois(
                    cluster         =df_cluster,
                    markers         =markers,
                    params          =params['PARAMS_EXTRACT'],
                    column_labels   ='marker',
                    unique_labels   =['FP', 'TP', 'FN', 'TN'],
                    filename        =filename,
                    random_seed     =None,
                    verbose         =True
                    )
        
#%% 
    # if 'mark_rois' in PROCESS :
        
    #     # Mark the ROIS with the prefix TN FN TP FP according to the clustering 
    #     # ---------------------------------------------------------------------
                
    #     df_rois, flag = bambird.mark_rois(
    #                     markers, 
    #                     dataset_csv   =csv_rois,            
    #                     verbose       =True
    #                     )
        
#%%
    # if 'unmark_rois' in PROCESS :
        
    #     # Unmark ROIS with the prefix TN FN TP FP  
    #     # ---------------------------------------------------------------------
                        
    #     df_rois, flag = bambird.unmark_rois(
    #                     dataset_csv =csv_rois,
    #                     verbose     =True
    #                     )