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
import pandas as pd
import matplotlib.pyplot as plt
plt.close("all")

import bambird

# %%
# Define constants
# ----------------

RANDOM_SEED = 1979
DIR_DATA        = Path('data/test')               
ANNOT_CSV_FILE  = 'manual_annotations.csv' 
ROIS_CSV_FILE   = "rois.csv" 
CONFIG_FILE     = 'config_article.yaml' 

# %%
if __name__ == '__main__':

    with open(CONFIG_FILE) as f:
        params = yaml.load(f, Loader=bambird.get_loader())
    
    # Name of the csv file with feaetures
    FEATURES_CSV_FILE = (
        'features_'
        + params['PARAMS_FEATURES']["SHAPE_RES"]
        + "_NFFT"
        + str(params['PARAMS_FEATURES']["NFFT"])
        + "_SR"
        + str(params['PARAMS_FEATURES']["SAMPLE_RATE"])
        + ".csv"
    )
#%%
    # Cluster ROIS
    # ---------------------------------------------------------------------

    # Set the variable dataset to be a csv file containing the dataframe
    dataset_features = DIR_DATA / FEATURES_CSV_FILE
    
    
    # dataset_features = DIR_DATA / 'features_a_virer.csv'
    dataset_features = pd.read_csv(dataset_features, sep=';')
    # dataset_features.drop('id', axis=1, inplace = True)

    # with dataframe or csv file
    df_cluster = bambird.find_cluster(
                            dataset     =dataset_features,
                            params      =params['PARAMS_CLUSTER'],
                            display     =True,
                            verbose     =True)

    # Evaluation of the clustering : precision and recall
    # ---------------------------------------------------------------------
    
    df_scores, p, r, f, markers = bambird.cluster_eval(
                            df_cluster          =df_cluster, 
                            path_to_csv_with_gt =DIR_DATA / ANNOT_CSV_FILE,
                            colname_label_gt    ='manual_label',
                            verbose             =True)
        