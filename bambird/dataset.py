#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Collection of functions to query metadata from xeno-canto and download the audio
to build a dataset.
"""
#
# Authors:  Felix MICHAUD   <felixmichaudlnhrdt@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#
# License: New BSD License

#%%
# general packages
import os
from pathlib import Path

# basic packages
import pandas as pd
import glob

# Scikit-Maad (ecoacoustics functions) package
import maad

#
from bambird import config as cfg

#%%

###############################################################################
def query_xc (species_list, 
           params=cfg.DEFAULT_PARAMS_XC,
           format_time=False,
           format_date=False,
           random_seed=cfg.RANDOM_SEED, 
           verbose=False):
    """
    Query metadata from Xeno-Canto website with audiofile depending on the search terms. 
    The audio recordings metadata are grouped and stored in a dataframe.

    Parameters
    ----------
    species_list : list
        List of scientific name of birds (e.g. 'Columba palumbus')
    
    params : list
        list of search terms to perform the query
        The main seach terms are :
        - q   : quality
        - cnt : country
        - len : length
        - area : continent (europe, africa, america, asia)
        see more here : https://www.xeno-canto.org/help/search
    format_time : boolean, optional
        Time in Xeno-Canto is not always present neither correctly formated. 
        If true, time will be correctly formated to be processed as DateTime 
        format. When formating is not possible, the row is dropped. 
        The default is False
    format_date : boolean, optional
        Date in Xeno-Canto is not always present neither correctly formated. 
        If true, rows with uncorrect format of date are dropped.
    verbose : boolean, optional
        Print messages during the execution of the function. The default is False.

    Returns
    -------
    df_dataset : pandas DataFrame
        Dataframe containing all the recordings metadata matching search terms
    """
    
    # separate the genus and the species from the scientific name
    gen = []
    sp = []
    for name in species_list:
        gen.append(name.rpartition(' ')[0])
        sp.append(name.rpartition(' ')[2])

    # add the genus and the species as query parameters
    df_query = pd.DataFrame()
    df_query[0] = gen
    df_query[1] = sp

    # add the other parameters
    if params is not None :
        idx = 2
        for param in params['PARAM_XC_LIST']:
            df_query[idx] = param
            idx += 1

    # Get recordings metadata corresponding to the query
    df_dataset = maad.util.xc_multi_query(df_query,
                                     max_nb_files=params['NUM_FILES'],
                                     format_time=format_date,
                                     format_date=format_date,
                                     random_seed=random_seed,
                                     verbose=verbose)

    # test if recordings are found
    if len(df_dataset) > 0 :
        # add scientific name and categories into the dataframe
        df_dataset.insert(4, 'categories', df_dataset['gen'] + ' ' + df_dataset['sp'])        
    else:  
        if verbose :
            print("No audio recording was found with the parameters: \n {}".format(params))

    
    return df_dataset

###############################################################################
def download_xc (df_dataset,
                rootdir, 
                dataset_name, 
                csv_filename="bam_metadata.csv",
                overwrite=False,
                verbose=True):
    """
    Download the audio files from Xeno-Canto based on the input dataframe
    It will create directories for each species if needed

    Parameters
    ----------
    df_dataset : pandas DataFrame
        Dataframe containing the selected recordings metadata to be downloaded.
        It could be the output of dataset_query, or a subset of this
        dataframe
    rootdir : string
        Path to the directory where the whole dataset will be saved
    dataset_name : string
        Name of the dataset that will be created as a parent directory . 
    csv_filename : string, optional
        Name of the csv file where the dataframe (df_dataset) will be saved. if
        the file already exists, data will be appended at the end of the csv file.
        The default is bam_metadata.csv
    overwrite : boolean, optional
        Test if the directory where the audio files will be downloaded already
        exists. if True, it will download the data in the directory anyway.
        Otherwise, if False, it will not download audio files.
    verbose : boolean, optional
        Print messages during the execution of the function. The default is False.

    Returns
    -------
    df_dataset : pandas DataFrame
        Dataframe containing all the selected audio recordings that were 
        resquested as input AND downloaded. If for some reasons, some of the 
        audio recordings cannot be downloaded, they do not appear in the output
        dataframe. 
    csv_fullfilename : string
        Full path to the csv file where the dataframe (df_dataset) was saved. 
        If the file already exists, data was appended at the end of the csv file.
        /!\ the csv file contains all the audio files that were recorded NOW and
        BEFPRE while the dataframe df_dataset contains ONLY the files that
        were recorded NOW.
    """
    # format rootdir as path
    rootdir = Path(rootdir)
    
    #-------------------------------------------------------------------
    # download all the audio files into a directory with a subdirectory for each
    # species
    df_dataset = maad.util.xc_download(df           =df_dataset,
                                       rootdir      =rootdir,
                                       dataset_name =dataset_name,
                                       overwrite    =overwrite,
                                       save_csv     =True,
                                       verbose      =verbose)
    
    #-------------------------------------------------------------------
    
    df_dataset['filename']  = df_dataset['fullfilename'].apply(os.path.basename)
    df_dataset['categories']   = df_dataset['fullfilename'].apply(lambda path : 
                                                                  Path(path).parts[-2])
        
    #--------------------------------------------------------------------------    
    # test if the csv file with all the metadata already exists and append the
    # df_dataset otherwise write a new csv file
    csv_fullfilename = rootdir / dataset_name / csv_filename
    if os.path.exists(csv_fullfilename):              
       # try to read the file and add the new rows (no duplicate)
        try :
                                                                   
            # remove from df_data the audio files that were already downloaded
            mask = df_dataset['filename'].isin(pd.read_csv(os.path.join(csv_fullfilename),
                                                           sep=';',
                                                           index_col='id')['filename'].unique().tolist())
            # append the new audio to the dataframe and save it
            df_dataset[~mask].to_csv(csv_fullfilename, 
                                     sep=";", 
                                     index=True,
                                     index_label='id',
                                     header = False,
                                     mode = 'a')                                                   
        except :
            pass                                                          
    else:
        # try to create a file and add a row corresponding to the index
        try :
            df_dataset.to_csv(csv_fullfilename, 
                              sep=";", 
                              index=True, 
                              index_label = 'id') 
        except :
            pass
   
    #--------------------------------------------------------------------------
    # display information about the downloaded audio files
    if verbose :        
        if len(df_dataset)>0 :
            print('*******************************************************')
            print('number of files : %2.0f' % len(df_dataset))
            print('number of categories : %2.0f' % len(df_dataset.categories.unique()))
            print('unique categories : {}'.format(df_dataset['categories'].unique()))
            print('*******************************************************')
        else :
            print('*** WARNING *** The dataframe is empty.')  
                        
    return df_dataset, csv_fullfilename

#%%
def grab_audio_to_df (path, 
                      audio_format, 
                      verbose=False) :
    """
    
    columns_name :
        First column name corresponds to full path to the filename
        Second column name correspond to the filename alone without the extension
    """
    
    # create a dataframe with all recordings in the directory
    filelist = glob.glob(os.path.join(path,
                                      '**/*.'+audio_format), 
                         recursive=True)
    
    df_dataset = pd.DataFrame()
    for file in filelist:
        categories = Path(file).parts[-2]
        iden = Path(file).parts[-1]
            
        df_dataset = df_dataset.append({
                                      'fullfilename':file,
                                      'filename'    :Path(file).parts[-1],
                                      'categories'  :categories,
                                      'id'          :iden},
                                    ignore_index=True)
        
    # set id as index
    df_dataset.set_index('id', inplace = True)
        
    if verbose :
        if len(df_dataset)>0 :
            print('*******************************************************')
            print('number of files : %2.0f' % len(df_dataset))
            print('number of categories : %2.0f' % len(df_dataset.categories.unique()))
            print('unique categories : {}'.format(df_dataset['categories'].unique()))
            print('*******************************************************')
        else :
            print('No {} audio file was found in {}'.format(audio_format,
                                                            path))    
    
    return df_dataset

#%%
def change_path (dataset_csv,
                 old_path,
                 new_path,
                 column_name,
                 verbose = False,
                 ) :
    
    # Read the csv
    df = pd.read_csv(dataset_csv)

    try:
        df[column_name] = df[column_name].str.replace(old_path, new_path)
        done = True
    except:
        done = False
        if verbose:
            print("**WARNING*** : No {} column is present in the dataframe".format(column_name))
        return done
    
    # save the dataframe with the new paths
    df.to_csv(dataset_csv, index=False)
    
    if verbose:
        print(' DONE ')
        print("Current path is {}".format (old_path)) 
        print(">>> New path is {}".format (new_path))
        
    return done
