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
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
plt.close("all")

import maad

# package bambird
bambird_path   = Path('../../bambird')
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
CONFIG_FILE     = 'config_soundscape.yaml' 

# dictionnary annotations
SOFTWARE        = 'audacity' # raven audacity
ANNOTATION_DIR  = DATASET_DIR / 'soundtypes_annotations'

# After the process, remove the audio that were saved during the process ?
CLEAN = True

# %%

def label_to_richness_abundance(folder,
                               labelformat  ='audacity',
                               extension    ='txt',
                               verbose      =True):
    
    """ 
    Build a dataframe with labels from annotations done with Audacity
     
    Parameters 
    ---------- 
    folder: string
        Path to the folder where the annotation files are saved.
        
    labelformat: string, default is 'audacity'
        Select how to decode the file containing the labels between 'audacity'
        or 'raven'.
         
    extension: string, default is 'txt'
        Extension of the filenames containing annotations 
     
    verbose: boolean, default is True
        Print messages in the console, in particular the files in the folder 
     
    Returns 
    ------- 
    df_label: pandas dataframe 
        dataframe with all the columns
        
    """
    
    # parse a directory in order to get a df with date and fullfilename
    df_labelfile = bambird.grab_audio_to_df (folder, extension, verbose)

    # concatenate dataframe with annotations : label + bounding box coordinate (tmin, fmin, tmax, fmax)
    df_label = pd.DataFrame()
    for index, row in df_labelfile.iterrows():
        
        # parse audacity file to get the label and the rois coordinates
        if labelformat == 'audacity' :
            dftemp = maad.util.read_audacity_annot(row['fullfilename'])
        elif labelformat == 'raven' :     
            dftemp = maad.util.read_raven_annot(row['fullfilename'])
        else :
            raise TypeError('Label format is not recognized')    
            
        # test if dftemp is empty which means that the csv is empty (no annotation)
        if dftemp.empty:
            dftemp = pd.DataFrame({'label': 'nan',
                                   'min_t': 0,
                                   'min_f': 0,
                                   'max_t': 0,
                                   'max_f': 0}, index=[0])
        
        try :     
            # remove white space at the end of string of the label columns
            dftemp.label = dftemp.label.str.rstrip()
            # create a serie
            serie_label = pd.Series()
            # add an item : File without extension .txt
            _, filename_with_ext = os.path.split(row['fullfilename'])
            serie_label['filename'] = filename_with_ext
            # Count each unique label in the file
            serie_label2 = dftemp['label'].value_counts()
            # concatenate series
            serie_label = serie_label.append(serie_label2)
            # convert serie into dataframe
            df_label_temp = pd.DataFrame(serie_label).T
            # add the new dataframe as a new row
            df_label = df_label.append(df_label_temp, sort=True)
        
        except :
            pass

    # change the position of the column file at the first position
    column_file = df_label.pop('filename')
    df_label.insert(0, 'filename', column_file)
    
    # remove the extension
    df_label['filename'] = df_label['filename'].apply(lambda file : file.split('.')[0])
    
    # add 2 columns 
    df_label['abundance'] = df_label.iloc[:,1:].sum(axis = 1)
    df_label['richness']  = df_label.iloc[:,1:].nunique(axis = 1)
    
    # keep only abundance and richness
    df_label = df_label[['filename','abundance', 'richness']]

    return df_label
    
# %%

# Load the configuration file    
params = cfg.load_config(CONFIG_FILE)

#%%    

# Grab audio
# -------------------
df = bambird.grab_audio_to_df(DATASET_DIR, 'wav')

#%% 

# Extract ROIS
# -------------------------------

# ROIS extraction of the full dataset
df_rois, csv_rois = bambird.multicpu_extract_rois(
                    dataset     = df,
                    params      = params['PARAMS_EXTRACT'],
                    save_path   = TEMP_DIR / ROIS_NAME,
                    overwrite   = True,
                    verbose     = True
                    )

# test if no ROI was found => break
if len(df_rois) == 0 :
    print ("No ROI was found")
    sys.exit()

#%%
# Compute features for each ROIS
# -------------------------------
   
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

df_cluster = pd.DataFrame()

for filename in df_features.filename.unique() :

    print(filename)
    
    df_cluster_single_file, _ = bambird.find_cluster(
                                    dataset = df_features[df_features['filename'] == filename],
                                    params  = params['PARAMS_CLUSTER'],
                                    display = False,
                                    verbose = True,
                                    )
    
    df_cluster = df_cluster.append(df_cluster_single_file)
           
#%%    
# Display the ROIS
# -------------------------------

# df_cluster["cluster_number"][df_cluster.groupby("cluster_number")["cluster_number"].transform('size') <= 2] = -1

# select a single file  
df_single = df_cluster[df_cluster["filename"] == df_cluster["filename"].sample(n=1).values[0]]
FILENAME = df_single['filename'].apply(lambda file : file.split('.')[0]).unique()[0]
FULLFILENAME = df_single['fullfilename'].values[0]

# parse a directory in order to get a df with date and fullfilename
df_labelfile = bambird.grab_audio_to_df (ANNOTATION_DIR, 'txt')
df_labelfile['file'] = df_labelfile['filename'].apply(lambda file : file.split('.')[0])
df_single_label = df_labelfile[df_labelfile['file'] ==  FILENAME]
if SOFTWARE == 'raven' :
    df_label = maad.util.read_raven_annot(df_single_label['fullfilename'].values[0])
    df_label.rename(columns = {'Begin Time (s)':'min_t',
                                    'End Time (s)':'max_t',
                                    'Low Freq (Hz)':'min_f',
                                    'High Freq (Hz)':'max_f'}, inplace = True)
else:
    df_label = maad.util.read_audacity_annot(df_single_label['fullfilename'].values[0]) 
    
df_label['filename'] = FILENAME + '.wav'
df_label['fullfilename'] = FULLFILENAME

# if abs_min_t not in the dataframe, create a new column and set the value to 0
if 'abs_min_t' not in df_label :
    df_label['abs_min_t'] = 0

bambird.overlay_rois(
                    cluster         = df_single,
                    params          = params['PARAMS_EXTRACT'],
                    column_labels   = 'cluster_number', #auto_label
                    unique_labels   = np.sort(df_single.cluster_number.unique()),
                    filename        = None, #'EBA_20141208_021800_right.wav', # EBA_20141210_021900_right
                    random_seed     = None,
                    verbose         = True
                    )


bambird.overlay_rois(
                    cluster         = df_label,
                    params          = params['PARAMS_EXTRACT'],
                    column_labels   = 'label', #auto_label
                    unique_labels   = np.sort(df_label.label.unique()),
                    filename        = None, #'EBA_20141208_021800_right.wav', # EBA_20141210_021900_right
                    random_seed     = None,
                    verbose         = True
                    )

#%%    
# Eval the clusters
# -------------------------------

######## delete the rows with noise (-1)
df_cluster = df_cluster[df_cluster["cluster_number"] != -1]

df_cluster_copy = df_cluster.copy()
# df_cluster_copy = df_cluster_copy[df_cluster_copy['categories']=='RSB']

########
df_label_auto = pd.DataFrame()
# add 2 columns 
df_label_auto['abundance'] = df_cluster_copy.groupby(["filename"])['cluster_number'].count()
df_label_auto['richness'] = df_cluster_copy.groupby(["filename"])['cluster_number'].nunique()
# release the index
df_label_auto.reset_index(inplace = True)
# remove the extension
df_label_auto['filename'] = df_label_auto['filename'].apply(lambda file : file.split('.')[0])

######## read csv
df_label_dico = label_to_richness_abundance(ANNOTATION_DIR, SOFTWARE)

######## rename columns
df_label_dico.rename(columns = {'abundance': 'acoustic abundance',
                                'richness' : 'acoustic richness'},
                     inplace=True) 
df_label_auto.rename(columns = {'abundance': 'cluster abundance',
                                'richness' : 'cluster richness'},
                     inplace=True) 

# sort on 'filename'
df_label_auto.sort_values('filename', inplace = True)
df_label_dico.sort_values('filename', inplace = True)

######## merge on 'filename'
df = df_label_dico.merge(df_label_auto, how='inner', on=['filename'])

# define Date as index, it removes the columns Date
df = df.set_index(['filename'])

# sort the dataframe by date
df.sort_index(inplace=True)

"""*****************************************************************
                        CORRELATION BETWEEN INDICES
******************************************************************"""
from scipy.stats import spearmanr, pearsonr

def spearmanr_pval(x,y):
        return spearmanr(x,y)[1]
def pearsonr_pval(x,y):
        return pearsonr(x,y)[1]
    
corr_matrix = df.corr('spearman')
pvalue_matrix = df.corr(method=spearmanr_pval)

pd.set_option('max_columns', corr_matrix.columns.size)
pd.set_option('display.expand_frame_repr', False)
corr_matrix

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(df['acoustic abundance'], df['cluster abundance'], alpha=0.66, s=30, color='royalblue')
ax2.scatter(df['acoustic richness'], df['cluster richness'],   alpha=0.75, s=30,  marker = 's' , color='tomato')
ax1.grid()
ax2.grid()
ax1.set_xlabel('acoustic abundance')
ax2.set_xlabel('acoustic richness')
ax1.set_ylabel("cluster abundance")
ax2.set_ylabel("cluster richness")
ax1.text(df['acoustic abundance'].quantile(0.06), 
         df['cluster abundance'].quantile(0.96), 
         "R = %.2f \np < 0.001" % corr_matrix['cluster abundance']['acoustic abundance'],
         )
ax2.text(df['acoustic richness'].quantile(0.06), 
         df['cluster richness'].quantile(0.96), 
         "R = %.2f \np < 0.001" % corr_matrix['cluster richness']['acoustic richness'])
fig.set_size_inches((6,3))
fig.tight_layout()


#%%    
# Remove files
# -------------------------------

if CLEAN :
    shutil.rmtree(TEMP_DIR, ignore_errors=True)