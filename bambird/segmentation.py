#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Collection of functions to segment the sound of interests (also called regions 
of interest (ROIs) in time and frequency.
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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Scipy
from scipy.io import wavfile

# audio package
import librosa

# Parallel processing packages
from functools import partial
from tqdm import tqdm
from concurrent import futures

# Scikit-Maad (ecoacoustics functions) package
import maad

#
from bambird import config as cfg
from bambird import grab_audio_to_df

#%%
###############################################################################

def _save_rois(
    chunk,
    fs,
    df_rois,
    save_path,
    margins=(0.02, 50),
    filter_order=5,
    display=False):
    """ 
    Save the rois as new wave files. 
    Margins around the bbox can  be applied around the bbox in order to 
    enlarge the save bbox in time and frequency
    
    Parameters
    ----------
    chunk: numpy array
        chunk of audio which a 1d vector of scalars to be saved
    fs: integer
        Sampling frequency of the audio
    df_rois: pandas dataframe
        dataframe with all the rois to be saved. Each roi is defined by limits
        in time and frequency that is used to cut the portion of audio that
        correspond to the roi.        
    margins: list of numbers, optional
        the first numbers corresponds to the +/- margins in time (s)
        the second numbers corresponds to the +/- margins in frequency (Hz)
        the default value is (0.02s, 50Hz)
    filter_order: integer, optional
        Define the order of the bandpass filter. The default value is 5
    display: boolean, optional
        If true, display the spectrogram of the roi that is saved
    
    Returns
    ------- 
    df_rois: pandas dataframe 
        returns the dataframe with a new column "fullfilename_ts" that contains
        the full path to the rois
    """

    fullfilename_ts_list = []
    
    # format save_path into Path
    save_path = Path(save_path)
    
    # Loop over the ROIs in the dataframe
    for index, row in df_rois.iterrows():

        # get the actual full path and create an new path to store
        # the ROIs as wav file
        full_path = Path(row["fullfilename"])
        # test if the save path is not a bird directory
        # if its already a bird directory, do not nest directory
        if save_path != full_path.parent:
            #new_path = os.path.join(save_path, full_path.parts[-2])
            new_path = save_path / full_path.parts[-2]
        else:
            new_path = save_path
        # create nested directories
        if os.path.exists(new_path) == False:
            new_path.mkdir(parents=True, exist_ok=True)
            
        # A) cut the raw audio in time domain
        idx_min = int((row["min_t"] - margins[0]) * fs)
        idx_max = int((row["max_t"] + margins[0]) * fs)
        # test if index are out of the boundary
        if idx_min < 0:
            idx_min = 0
        if idx_max >= len(chunk):
            idx_max = len(chunk) - 1
        # cut in time
        chunk_rois = chunk[idx_min:idx_max]

        # B) cut the audio in the frequency domain
        fcut_min = row["min_f"] - margins[1]
        fcut_max = row["max_f"] + margins[1]
        # test if frequencies are out of the boundary
        # fcut should be >0 and <fs/2
        if fcut_min <= 0:
            fcut_min = row["min_f"]
        if fcut_max > fs / 2:
            fcut_max = row["max_f"]
        # cut in frequency
        chunk_rois = maad.sound.select_bandwidth(
            chunk_rois,
            fs,
            fcut=[fcut_min, fcut_max],
            forder=filter_order,
            fname="butter",
            ftype="bandpass",
        )

        # Let's display the spectrogram of the ROI
        if display:
            # compute the spectrogram
            X, _, _, ext = maad.sound.spectrogram(
                chunk_rois, fs, nperseg=1024, noverlap=1024 // 2
            )
            kwargs = {"vmax": np.max(X)}
            kwargs.update({"vmin": np.min(X)})
            kwargs.update({"extent": ext})
            kwargs.update({"figsize": (1, 2.5)})
            maad.util.plot_spectrogram(
                X, extent=ext, title="saved spectrogram", 
                interpolation=None, now=True
            )

        # C) Write the chunk as a wav file
        # add an index to the filename in order to save unique
        # filename => filename_ts (for time stamp)
        fullfilename_ts = str(new_path) + "/" + row["filename_ts"]
        wavfile.write(fullfilename_ts, fs, np.float32(chunk_rois))

        fullfilename_ts_list.append(fullfilename_ts)

    # # add the full path for the chunk in the dataframe
    df_rois.insert(1, "fullfilename_ts", fullfilename_ts_list)

    return df_rois



###############################################################################
def single_file_extract_rois(
    audio_path,
    fun,
    params=cfg.PARAMS['PARAMS_EXTRACT'],
    save_path=None,
    display=False,
    verbose=False):
    """ Extract all Rois in the audio file
    Parameters
    ----------
    audio_path : TYPE
        DESCRIPTION.
    fun : function
        name of the function that is called to segment the rois
    params : dictionnary
        contains all the parameters to extract the rois 
    save_path : string, default is None
        Path to the directory where the segmented rois will be saved    
    display : TYPE, optional
        DESCRIPTION. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.
    Returns
    -------
    df_rois : TYPE
        DESCRIPTION.
    """

    plt.style.use("default")

    # extract audio filename from fullfilename
    path, filename_with_ext = os.path.split(audio_path)
    filename = filename_with_ext
    # extract species and categories from fullfilename 
    try :
        _, species = os.path.split(path)
        categories = ((species.split(" ", 1)[0][0:3]).lower() + 
                (species.split(" ", 1)[1][0:3]).lower())
    except :
        _, species = os.path.split(path)
        categories = species
    
    try:
        # 1. load audio
        sig, sr = librosa.load(
            audio_path, sr=params["SAMPLE_RATE"], duration=params["AUDIO_DURATION"])

        # remove DC value
        sig -= np.mean(sig)

        # 2. bandpass filter around birds frequencies
        fcut_max = min(params["HIGH_FREQ"], params["SAMPLE_RATE"] // 2 - 1)
        fcut_min = params["LOW_FREQ"]
        sig = maad.sound.select_bandwidth(
            sig,
            params["SAMPLE_RATE"],
            fcut=[fcut_min, fcut_max],
            forder=params["BUTTER_ORDER"],
            fname="butter",
            ftype="bandpass")

        # Split signal into CHUNK_DURATION second chunks
        # Just like we did before (well, this could actually be a seperate function)
        # We also adjust the overlap (OVLP, ratio) between each chunk
        sig_splits = []
        n_points_per_chunk = int(params["CHUNK_DURATION"]* params["SAMPLE_RATE"])
        for i in range(0,len(sig), int(n_points_per_chunk * (1 - params["OVLP"]))):

            split = sig[i: i + n_points_per_chunk]

            # End of signal?
            if len(split) < n_points_per_chunk :
                break

            sig_splits.append(split)

        # print outputs
        if verbose:
            print("\n================================================================")
            print("CATEGORIES : {}".format(categories))
            print("NUMBER OF CHUNKS : {}".format(len(sig_splits)))

        # if no chunk was found, pass
        if sig_splits == 0:
            pass
        else:

            # Find ROIs and compute features of each chunk and add them into df_rois
            seconds = 0
            df_rois = pd.DataFrame()
            for chunk in sig_splits:

                # call the function fun
                current_df_rois = fun(sig=chunk, 
                                    params=params, 
                                    display=display, 
                                    verbose=verbose,
                                    kwarg={'suptitle': filename + " " + categories})
                
                # test if at least 1 ROI was found
                if len(current_df_rois) > 0 :
                    
                    # Add global information about the file and categories for 
                    # all rois (rows)
                    filename_ts = [
                        filename.split(".",1)[0] + "_" +
                        str(seconds) + "s_" + "{:02d}".format(index) + ".wav"
                        for index, _ in current_df_rois.iterrows()
                    ]
                    
                    current_df_rois.insert(0, "filename_ts", filename_ts)
                    current_df_rois.insert(1, "filename", filename)
                    current_df_rois.insert(2, "fullfilename", audio_path)
                    current_df_rois.insert(3, "species", species)
                    current_df_rois.insert(4, "categories", categories)
                    current_df_rois.insert(5, "abs_min_t", seconds)
                    current_df_rois.insert(4, "id", filename.split('.',1)[0][2:])
                    
                    # Save ROIs as raw audio file
                    if save_path is not None:                        
                        if len(current_df_rois) > 0:
                            current_df_rois = _save_rois(
                                chunk,
                                params["SAMPLE_RATE"],
                                current_df_rois,
                                save_path=save_path,
                                margins=(params["MARGIN_T"],
                                         params["MARGIN_F"]),
                                filter_order=params["FILTER_ORDER"],
                                display=False,
                            )
          
                    # Add current_df_rois to df_rois
                    if len(df_rois) > 0:
                        df_rois = df_rois.append(
                            current_df_rois, ignore_index=True)
                    else:
                        df_rois = current_df_rois

                # Keep track of the end time of each chunk
                # it there is an overlap between chunks, take it into account
                seconds += params["CHUNK_DURATION"] * (1 - params["OVLP"])

    except Exception as e:
        if verbose:
            print("\nERROR : " + str(e))
        df_rois = pd.DataFrame()

    return df_rois

###############################################################################
def multicpu_extract_rois(
    dataset, 
    params=cfg.PARAMS['PARAMS_EXTRACT'],
    save_path=None,
    save_csv_filename='rois.csv',
    overwrite=False,
    nb_cpu=None,
    verbose=True):
    """ 
    Extract all Rois in the dataset (multiple audio files)
    
    Parameters
    ----------
    dataset : string or pandas dataframe
        if it's a string it should be either a directory where are the audio
        files to process or a full path to a csv file containing a column
        "filename" and a column "fullfilename" with the full path to the audio
        files to process
        if it's a dataframe, the dataframe should contain a column
        "filename" and a column "fullfilename" with the full path to the audio
        files to process. This dataframe can be obtained by called the function
        grab_audio_to_df        
    params : dictionnary, optioanl
        contains all the parameters to extract the rois 
    save_path : string, default is None
        Path to the directory where the segmented rois will be saved    
    save_csv_filename: string, optional
        csv filename that contains all the rois that will be saved. The default
        is rois.csv
    overwrite : boolean, optional
        if a directory already exists with the rois, if false, the process is 
        aborted, if true, new rois will eventually be added in the directory and
        in the csv file.
    nb_cpu : integer, optional
        number of cpus used to segment the rois. The default is None which means
        that all cpus will be used
    verbose : boolean, optional
        if true, print information. The default is False.
        
    Returns
    -------
    df_rois_sorted : pandas dataframe
        dataframe containing of the rois found in the audio. Each roi is
        characterized by a bounding box (min_f max_f, min_t max_t)
    csv_fullfilename : string
        full path the csv file with all the rois that were segmented. if the file
        already exists, the new rois will be appended to the file.
        
    See
    ---
    grab_audio_to_df
    """
    
    if verbose :
        print('======================= EXTRACT ROIS =========================\n')

    # if Dataset is a directory : 
    # > all audio files in the directory will be processed.
    #-----------------------------------------------------
    if isinstance(dataset, pd.DataFrame) == False :
        # test if dataset is a path to a directory
        #-----------------------------------------
        if os.path.isdir(dataset):
            
            # format dataset to Path
            dataset = Path(dataset)

            # create a dataframe with all recordings in the directory
            #--------------------------------------------------------
            df_data = grab_audio_to_df (path            =dataset, 
                                        audio_format    ='mp3',
                                        verbose         =verbose)
            
            # set default save_path and save_filename
            #----------------------------------------
            if save_path is None:
                if (dataset[-1] == "/") or (dataset[-1] == "/"):
                    dataset = dataset[:-1]
                save_path = str(dataset) + "_ROIS"
            
        # test if dataset_path is a valid csv file
        #----------------------------------------
        elif os.path.isfile(dataset):
            # load the data from the csv
            df_data = pd.read_csv(dataset, sep=';')
            
            # set default save_path and save_filename
            if save_path is None:
                save_path = os.path.dirname(dataset) + "_ROIS"
                
                
    # if dataset is a dataframe : 
    # > read the dataframe       
    #--------------------------- 
    elif isinstance(dataset, pd.DataFrame): 
        df_data = dataset.copy()
        
        # set default save_path and save_filename
        if save_path is None:
            save_path = str(Path(df_data['fullfilename'].iloc[0]).parent.parent) + "_ROIS"
            
    else:
        raise Exception(
            "WARNING: dataset must be a valid path to a directory, a csv file"
            + "or a dataframe"
        )

    # Check if the output directory already exists
    #---------------------------------------------
    dir_exist = os.path.exists(save_path) 
    if (dir_exist == False) or ((dir_exist == True) and (overwrite == True)) : 
        if (dir_exist == True) and (overwrite == True):
            if verbose:
                print(("The directory {} already exists" +
                      " and will be overwritten").format(save_path))   
        
        try :
            # load the dataframe with all ROIs already extracted
            #---------------------------------------------------
            df_rois = pd.read_csv(save_path / save_csv_filename, sep=';')
            # create a mask to select or not the audio files that were already segmented
            mask = df_data['filename'].isin(df_rois['filename'].unique().tolist())
            
        except :
            # create an empty dataframe. It will contain all ROIs found for each
            # audio file in the directory
            df_rois = pd.DataFrame()  
            
            # create a mask full of false
            mask = np.zeros(len(df_data), bool)
        
        INITIAL_NUM_ROIS = len(df_rois)
                
        # Extract ROIs using multicpu  
        #-----------------------------
        # test if the dataframe contains files to segment
        if len(df_data[~mask])>0 :
            
            if verbose :
                print('Composition of the dataset : ')
                print('   -number of files : %2.0f' % len(df_data[~mask]))
                print('   -number of categories : %2.0f' % len(df_data[~mask].categories.unique()))
                print('   -unique categories : {}'.format(df_data[~mask]['categories'].unique()))
        
            # Number of CPU used for the calculation. By default, set to all available
            # CPUs
            if nb_cpu is None:
                nb_cpu = os.cpu_count()
                
            # define a new function with fixed parameters to give to the multicpu pool 
            #-------------------------------------------------------------------------        
            
            # Print the characteristics of the function used to segment the files
            if verbose :
                print(params['FUNC'])
            
            multicpu_func = partial(
                single_file_extract_rois,
                fun=params['FUNC'],
                params=params,
                save_path=save_path,
                display=False,
                verbose=False,
            )
        
            # Multicpu process
            #-------------------
            with tqdm(total=len(df_data[~mask])) as pbar:
                with futures.ProcessPoolExecutor(max_workers=nb_cpu-1) as pool:
                    for df_rois_temp in pool.map(
                        multicpu_func, df_data[~mask]["fullfilename"].to_list()
                    ):
                        pbar.update(1)
                        df_rois = df_rois.append(df_rois_temp)
            
            # sort filename for each categories
            #---------------------------------
            df_rois_sorted = pd.DataFrame()
            for categories in df_rois["categories"].unique():
                df_rois_sorted = df_rois_sorted.append(
                    df_rois[df_rois["categories"] == categories].sort_index()
                )
                        
            if verbose :
               print('\n')
               print(('{} new ROIs added in {}').format(len(df_rois_sorted)-INITIAL_NUM_ROIS,
                                                        save_path))
               
            # save rois
            #-------------------------------
            if save_csv_filename is not None :
                # Set filename_ts to be the index before saving the dataframe
                try:
                    df_rois_sorted.set_index(['filename_ts'], inplace=True)
                except:
                    pass  
                
                # test if the directory exists if not, create it 
                if os.path.exists(save_path) == False:
                    save_path.mkdir(parents=True, exist_ok=True)
                
                # save and append dataframe 
                csv_fullfilename = save_path / save_csv_filename
                df_rois_sorted.to_csv(csv_fullfilename, 
                                      sep=';', 
                                      header=True)
                # reset index
                df_rois_sorted.reset_index(inplace=True)
        
        else:
            # reset the index
            try:
                df_rois.reset_index('filename_ts', inplace = True)
            except:
                pass  
            
            # remove from df_data the audio files that were already segmented
            mask = df_rois['filename'].isin(df_data['filename'].unique().tolist())
            df_rois_sorted = df_rois[mask]
            csv_fullfilename = save_path / save_csv_filename
            
            if verbose:
                print("No audio file needs to be segmented")
                print(">>> EXTRACTION PROCESS ABORTED <<<")
            
    # The directory already exists
    #-------------------------------      
    else:
        # Read the already existed file
        csv_fullfilename = save_path / save_csv_filename
        
        try :
            df_rois_sorted = pd.read_csv(csv_fullfilename, sep=";")
        except :
            df_rois_sorted = pd.DataFrame()
            
            if verbose :
                print(('***WARNING** The csv file {0} does not exist in the directory {1}').format(
                    save_csv_filename, save_path))
                
        if verbose:
            print("The directory {} already exists".format(save_path))
            print(">>> EXTRACTION PROCESS ABORTED <<<")

    return df_rois_sorted, csv_fullfilename

