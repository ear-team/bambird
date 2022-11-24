#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Collection of functions to compute the features for each sound of interest also
called regions of interest (ROIs)
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

# audio package
import librosa

# Parallel processing packages
from functools import partial
from tqdm import tqdm
from concurrent import futures

# Scikit-Maad (ecoacoustics functions) package
import maad

#
from bamxc import config as cfg
from bamxc import grab_audio_to_df


#%%
""" ===========================================================================

                    Features extraction : process the whole dataset

============================================================================"""

###############################################################################
def compute_features(
    audio_fullfilename,
    params=cfg.DEFAULT_PARAMS_FEATURES,
    display=False,
    verbose=False):
    
    # Custom Functions
    #
    # 1.   Load an audio
    # 2.   Bandpass the audio
    # 3.   Compute the spectrogram
    # 4.   Convert to dB
    # 5.   Remove background noise
    # 6.   Compute the features shape of the ROIs 
    # 7.   Compute the centroid and bandwidth of the ROIs
    # 8.   Add the features into a dataframe
    #

    plt.style.use("default")

    # extract audio filename and species from fullfilename
    path, filename_with_ext = os.path.split(audio_fullfilename)
    filename = filename_with_ext
    try :
        _, species = os.path.split(path)
        code = ((species.split(" ", 1)[0][0:3]).lower() + 
                (species.split(" ", 1)[1][0:3]).lower())
    except :
        _, species = os.path.split(path)
        code = species
    
    if verbose:
        print("file XC {}".format(filename))

    try:
        # 1. load audio
        sig, sr = librosa.load(audio_fullfilename, sr=params["SAMPLE_RATE"])

        # 2. bandpass filter around birds frequencies
        fcut_max = min(params["HIGH_FREQ"], params["SAMPLE_RATE"] // 2 - 1)
        fcut_min = params["LOW_FREQ"]
        sig = maad.sound.select_bandwidth(sig,
                                    params["SAMPLE_RATE"],
                                    fcut=[fcut_min, fcut_max],
                                    forder=params["BUTTER_ORDER"],
                                    fname="butter",
                                    ftype="bandpass")

        # 3. compute the spectrogram
        Sxx, tn, fn, ext = maad.sound.spectrogram(sig,
                                             params["SAMPLE_RATE"],
                                             nperseg=params["NFFT"],
                                             noverlap=params["NFFT"] // 2)

        # set to zero the DC value (= the first row of the spectrogram)
        Sxx[0, :] = 0
        Sxx[1, :] = 0

        if display:
            # prepare 4 figures in a row
            fig, ax = plt.subplots(2, 2, figsize=(10, 5))
            # plot the spectrogram into a bigger plot full of zeros with fixed
            # size in order to compare the spectrograms
            WIDTH_T = 20  # in s. Fix the width of the spectrogram.
            # width in pixels
            width_x = round(WIDTH_T / (tn[1] - tn[0]))
            # zero padding before and after the spectrogram if the duration time
            # is below width_x
            if int(round((width_x - Sxx.shape[1]) / 2)) > 0 :
                margin = np.zeros((Sxx.shape[0], 
                                  int(round((width_x - Sxx.shape[1]) / 2))), int)
                Sxx_to_display = np.concatenate((margin, Sxx, margin), axis=1)
            else :
                Sxx_to_display = Sxx
                
            maad.util.plot_spectrogram(Sxx_to_display,
                                  extent=[0, WIDTH_T, ext[2], ext[3]],
                                  ax=ax[0, 0],
                                  title="original ROI",
                                  now=False)

        # 4. convert to dB
        Sxx_dB = maad.util.power2dB(Sxx, db_range=96) + 96

        # 5. Clean spectrogram : remove background
        Sxx_clean_dB, _, _ = maad.sound.remove_background(Sxx_dB)
        Sxx_clean_dB = Sxx_clean_dB * (Sxx_clean_dB > 6)
        
        # Sxx_clean_dB = Sxx_dB
        

        if display:
            maad.util.plot_spectrogram(
                np.concatenate((margin, Sxx_clean_dB, margin), axis=1),
                # [left, right, bottom, top]
                extent=[0, WIDTH_T, ext[2], ext[3]],
                log_scale=False,
                ax=ax[0, 1],
                title="cleaned ROI",
                now=False)

        # initialize a dataframe with ROIs with the coordinate of the
        # bounding box
        # !!! it does not work when using the original bbox values found during
        # the segmentation. This might be due to the different resolution which
        # creates pixel shift.

        # test if we can find 0 pixels. If not, set the min_y to 0 and max_y to
        # the vertical size of the spectrogram
        if (len(np.where(maad.sound.avg_amplitude_spectro(Sxx_clean_dB) != 0)[0])) <= 2:
            min_y = 0
            max_y = Sxx_clean_dB.shape[1]
        else:
            min_y = np.where(maad.sound.avg_amplitude_spectro(Sxx_clean_dB) != 0)[0][2]  # 2 to avoid to take indice 0 or 1
            max_y = np.where(maad.sound.avg_amplitude_spectro(Sxx_clean_dB) != 0)[0][-1]

        df_rois_for_shape = pd.DataFrame([[min_y, 0, max_y, Sxx.shape[1] - 1]], 
                                         columns=["min_y", "min_x", "max_y", "max_x"])

        # 6. Compute acoustic features (MAAD)
        df_shape, params_shape = maad.features.shape_features(Sxx_clean_dB, 
                                                         resolution=params["SHAPE_RES"], 
                                                         rois=df_rois_for_shape)

        if display:
            maad.util.plot_shape(df_shape, params_shape, ax=ax[1, 0])

        # Keep columns ['min_y', 'min_x', 'max_y', 'max_x'] in case of these
        # features are not in the original dataset. If there are already
        # in the original dataset, these columns will be deleted and we keep
        # the original columns ['min_y', 'min_x', 'max_y', 'max_x'] found during
        # the extraction of the ROIs
        df_shape = maad.util.format_features(df_shape, tn, fn)

        # 7. Compute the centroid (t,f), the bandwidth and the duration of the ROI

        # initialize the dataframe df_rois with duration and centroid from the
        # size of the spectrogram
        df_rois = pd.DataFrame([[tn[-1], tn[-1] / 2]], columns=["duration_t", "centroid_t"])

        # compute the mean spectrum and smooth the spectrum with a running mean
        Sxx_clean = maad.util.dB2power(Sxx_clean_dB)
        mean_spectrum = maad.util.running_mean(maad.sound.avg_power_spectro(Sxx_clean), N=35)

        # Convert the spectrum in dB
        mean_spectrum_dB = maad.util.power2dB(mean_spectrum)

        # find the frequency position and amplitude corresponding to the
        # frequency peak
        f_peak = fn[np.argmax(mean_spectrum_dB)]
        f_peak_amplitude_dB = np.max(mean_spectrum_dB)

        # find the frequencies (- and +) at -6dB
        index = [i for i, x in enumerate(mean_spectrum_dB) if x > (f_peak_amplitude_dB-6)]
        f_peak_minus_6dB = fn[index[0]]
        f_peak_plus_6dB = fn[index[-1]]

        # add the values to the dataframe
        df_rois["peak_f"] = f_peak
        df_rois["centroid_f"] = (f_peak_plus_6dB + f_peak_minus_6dB) / 2
        df_rois["bandwidth_f"] = f_peak_plus_6dB - f_peak_minus_6dB
        df_rois["bandwidth_min_f"] = f_peak_minus_6dB
        df_rois["bandwidth_max_f"] = f_peak_plus_6dB
        
        if display:
            maad.util.plot_spectrum(mean_spectrum_dB, fn, ax=ax[1, 1])
            ax[1, 1].axvline(x=f_peak_minus_6dB, color="red", linestyle="dotted", linewidth=2)
            ax[1, 1].axvline(x=f_peak_plus_6dB, color="red", linestyle="dotted", linewidth=2)
            ax[1, 1].plot([f_peak], [f_peak_amplitude_dB], "r+")
            ax[1, 1].plot([(f_peak_plus_6dB + f_peak_minus_6dB) /2],[f_peak_amplitude_dB], "bo")
            fig.suptitle(filename + " " + code)
            fig.tight_layout()

        # add global information about the file and species for all rois (rows)
        df_rois.insert(0, "filename_ts", filename)
        df_rois.insert(1, "fullfilename_ts", audio_fullfilename)

        # 10. concat df_rois and df_shape
        df_features = pd.concat([df_rois, df_shape], axis=1)

    except Exception as e:
        print("\n" + str(e))
        df_features = pd.DataFrame()

    return df_features

###############################################################################
def multicpu_compute_features(
                dataset, 
                params=cfg.DEFAULT_PARAMS_FEATURES,
                save_path=None,
                save_csv_filename=None,
                nb_cpu=None,
                overwrite=False,
                verbose=True):

    if verbose :
        print('\n')
        print('====================== COMPUTE FEATURES ======================\n')
        
    # if dataset is a directory : 
    # > all audio files in the directory will be processed. 
    #-------------------------------------------------------------------------
    # test if dataset is a valid path with audio files
    if isinstance(dataset, pd.DataFrame) == False:
        
        # format dataset into Path
        dataset = Path(dataset)
        
        # test if it's a directory
        if os.path.isdir(dataset):
    
            # create a dataframe with all recordings in the directory
            df_rois = grab_audio_to_df(path         =dataset, 
                                        audio_format ='wav', 
                                        verbose      =verbose)
            
            # change the default name of the column fullfilename when using
            # grab_audio_to_df
            df_rois = df_rois.rename(columns={"fullfilename":"fullfilename_ts",
                                              "filename":"filename_ts"})
            
            # retrieve the origianl filename from filename_ts
            df_rois['filename'] = df_rois['filename_ts'].apply(lambda filename_ts : 
                                                                  (filename_ts.split("_", 1)[0])+'.mp3')
            
            # Default directory to save the dataframe with all features
            #----------------------------------------------------------
            if save_path is None :
                save_path = dataset
                
        # test if dataset is a valid csv file
        #------------------------------------
        elif os.path.isfile(dataset):
            # load the data from the csv
            df_rois = pd.read_csv(dataset, sep = ';')
            
            # # change the default name of the column fullfilename
            # df_rois = df_rois.rename(columns={"fullfilename":"fullfilename_ts",
            #                                   "filename":"filename_ts"})
            
            # Default directory to save the dataframe with all features
            #----------------------------------------------------------
            if save_path is None :
                save_path = dataset.parent
            
    # if dataset is a dataframe : 
    # > read the dataframe       
    #-------------------------------------------------------------------------
    elif isinstance(dataset, pd.DataFrame): 
        df_rois = dataset.copy()
        
        # Default directory to save the dataframe with all features
        #----------------------------------------------------------
        if save_path is None:
            save_path = str(Path(df_rois['fullfilename_ts'].iloc[0]).parent.parent)

    else:
        raise Exception(
            "WARNING: dataset must be a valid path to a directory, a csv file"
            + "or a dataframe"
        )
              
    # Default name of the csv file with features
    #-------------------------------------------------------------------------
    if save_csv_filename is None :
        save_csv_filename = (
            'features_'
            + params["SHAPE_RES"]
            + "_NFFT"
            + str(params["NFFT"])
            + "_SR"
            + str(params["SAMPLE_RATE"])
            + ".csv"
        )
                
    # Test if the csv with features already exists and if we want to overwrite it.
    #-------------------------------------------------------------------------
    # format save_path into Path
    save_path = Path(save_path)
    file_exist = (save_path / save_csv_filename).exists()
    if (file_exist == False) or ((file_exist == True) and (overwrite == True)) : 
        if ((file_exist == True) and (overwrite == True))  :
            if verbose :
                print(('{} already exists. \n'+
                      '***BE CAREFULL*** csv file will be overwritten').format(save_csv_filename)) 
                
        #----------------------------------------------------------------------
        try :
            # load the dataframe with all ROIs already extracted
            df_features = pd.read_csv(save_path / save_csv_filename, 
                                      sep=';')
            # create a mask to select or not the audio files that were already segmented
            mask = df_rois['filename'].isin(df_features['filename'].unique().tolist())
            
        except :
            # create an empty dataframe. It will contain all ROIs found for each
            # audio file in the directory
            df_features = pd.DataFrame()  
            
            # create a mask full of false
            mask = np.zeros(len(df_rois), bool)
        
        INITIAL_NUM_ROIS = len(df_features)
    
        # Compute features using multicpu  
        #---------------------------------
        # test if the dataframe contains files to compute the features
        if len(df_rois[~mask]) > 0 :
            
            if verbose :
                print('Composition of the dataset : ')
                print('   -number of files : %2.0f' % len(df_rois[~mask]))
                print('   -number of species : %2.0f' % len(df_rois[~mask].species.unique()))
                print('   -unique species code : {}'.format(df_rois[~mask]['code'].unique()))
            
            # Number of CPU used for the calculation. By default, set to all available
            # CPUs
            if nb_cpu is None:
                nb_cpu = os.cpu_count()
        
            # define a new function with fixed parameters to give to the multicpu pool
            multicpu_func = partial(
                compute_features,
                params=params,
                display=False,
                verbose=False,
            )
        
            with tqdm(total=len(df_rois[~mask])) as pbar:
                with futures.ProcessPoolExecutor(max_workers=nb_cpu-1) as pool:
                    for df_features_temp in pool.map(
                        multicpu_func, df_rois[~mask]["fullfilename_ts"].to_list()
                    ):
                        pbar.update(1)
                        df_features = df_features.append(df_features_temp)
        
            # Merge the result df_features with the df_rois
            #----------------------------------------------
            # Find the columns that are also in the first DataFrame
            # For instance, if columns ['min_y', 'min_x', 'max_y', 'max_x'] are in both
            # dataframe, we will keep the original value from df_data found during
            # the extraction of the ROIs
            
            df_rois.set_index(["filename_ts"], inplace=True)
            df_features.set_index(["filename_ts"], inplace=True)
            drop_col = (df_features.columns & df_rois.columns).tolist()
        
            # Merge the DataFrames based on the index and after dropping the columns
            # that are duplicated
            df_features = pd.merge(
                df_features.drop(drop_col, axis=1),
                df_rois,
                how="outer",
                left_index=True,
                right_index=True,
            )
            
            # sort filename for each species
            #-------------------------------
            df_features_sorted = pd.DataFrame()
            for code in df_features["code"].unique():
                df_features_sorted = df_features_sorted.append(
                    df_features[df_features["code"] == code].sort_index()
                )
           
            if verbose :
                print('\n')
                print(('Features were extracted for'
                       +' {} ROIS audio files').format(len(df_features_sorted)- INITIAL_NUM_ROIS))
             
            # save features
            #-------------------------------
            if save_csv_filename is not None :
                # Set filename_ts to be the index before saving the dataframe
                try:
                    df_features_sorted.set_index(['filename_ts'], inplace=True)
                except:
                    pass  
                # save and append dataframe 
                csv_fullfilename = save_path / save_csv_filename
                df_features_sorted.to_csv(csv_fullfilename, 
                                          sep=';', 
                                          header=True)
                # reset index
                df_features_sorted.reset_index(inplace=True)
  
        else :        
            # reset the index
            try:
                df_features.reset_index('filename_ts', inplace = True)
            except:
                pass       
                    
            # remove from df_rois the audio files that were already computed
            mask = df_features['filename'].isin(df_rois['filename'].unique().tolist())
            df_features_sorted = df_features[mask]
            csv_fullfilename = save_path / save_csv_filename
            
            if verbose :
                print("No feature needs to be computed")
                print(">>> FEATURES COMPUTING PROCESS ABORTED <<<")
                
    # The csv file already exists
    #-------------------------------      
    else:
        # Read the already existed file
        csv_fullfilename =save_path / save_csv_filename
        
        try :
            df_features_sorted = pd.read_csv(csv_fullfilename, sep=";")
        except :
            df_features_sorted = pd.DataFrame()
            
            if verbose :
                print(('***WARNING** The csv file {0} does not exist in the directory where are the ROIS. \n' +
                      '====> Please, delete the directory {1} and restart the extraction process').format
                      (save_csv_filename, save_path))
                
        if verbose:
            print("The features file {} already exists".format(save_csv_filename))
            print(">>> EXTRACTION PROCESS ABORTED <<<")
            
    return df_features_sorted, csv_fullfilename
