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
DIR_DATA        = Path('./data/test')               
ANNOT_CSV_FILE  = 'manual_annotations.csv' 
BIRDNET_CSV_FILE= "birdnet_annotations.csv" 
CONFIG_FILE     = 'config_article.yaml' 

# %%
if __name__ == '__main__':

    with open(CONFIG_FILE) as f:
        params = yaml.load(f, Loader=bambird.get_loader())
    
# %%         
         
    # BirdNET on ROIS 
    # ---------------------------------------------------------------------
    # Load the dataframe with the result from BirdNET
    df_cluster = pd.read_csv(DIR_DATA / BIRDNET_CSV_FILE, sep=';')
        
    # Evaluation of the clustering : precision and recall
    # ---------------------------------------------------------------------
    df_scores, p, r, f, markers = bambird.cluster_eval(
                            df_cluster, 
                            path_to_csv_with_gt=DIR_DATA / ANNOT_CSV_FILE,
                            colname_label='birdnet_label',
                            verbose=True)
    
       
