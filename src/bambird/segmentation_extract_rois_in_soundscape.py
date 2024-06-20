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

# basic packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# scikit-image package
from skimage import measure
from skimage.morphology import closing

from scipy import ndimage 

# Scikit-Maad (ecoacoustics functions) package
import maad
from maad.util import mean_dB, add_dB, power2dB, dB2power

# import bamx
from bambird import config as cfg


PARAMS_EXTRACT = {'SAMPLE_RATE': 48000,
                 'LOW_FREQ': 250,
                 'HIGH_FREQ': 12000,
                 'BUTTER_ORDER': 1,
                 'AUDIO_DURATION': 30,
                 'CHUNK_DURATION': 10,
                 'OVLP': 0,
                 'MODE_RMBCKG': 'median',
                 'N_RUNNING_MEAN': 10,
                 'NFFT': 1024,
                 'SEED_LEVEL': 26,
                 'LOW_LEVEL': 10,
                 'MAX_RATIO_YX': 7,
                 'MIN_EVENT_DUR': 0.1,
                 'MAX_EVENT_DUR': 60,
                 'FUSION_ROIS': (0.05, 100),
                 'MIN_FREQ_BW': 0.2,
                 'MAX_FREQ_BW': None,
                 'REMOVE_ROIS_FMIN_LIM' : 50,
                 'REMOVE_ROIS_FMAX_LIM' : None,
                 'REMOVE_RAIN': True,
                 'MARGIN_T': 0.1,
                 'MARGIN_F': 250,
                 'FILTER_ORDER': 5}


#%%
###############################################################################
def _centroid_features(Sxx, rois=None, im_rois=None): 
    """ 
    Computes intensity centroid of a spectrogram. If regions of interest 
    ``rois`` are provided, the centroid is computed for each region.
    
    Parameters 
    ---------- 
    Sxx :  2D array 
        Spectrogram in dB scale
    rois: pandas DataFrame, default is None 
        Regions of interest where descriptors will be computed. Array must  
        have a valid input format with column names: ``min_t``, ``min_f``, 
        ``max_t``, and ``max_f``. Use the function ``maad.util.format_features``
        before using centroid_features to format of the ``rois`` DataFrame 
        correctly.
    im_rois: 2d ndarray 
        image with labels as values
             
    Returns 
    ------- 
    centroid: pandas DataFrame 
        Centroid of each region of interest.
        
    See Also
    --------
    maad.features.shape_features, maad.util.overlay_rois, 
    maad.util.format_features
 
    Examples
    --------
 
    Get centroid from the whole power spectrogram 
 
    >>> from maad.sound import load, spectrogram
    >>> from maad.features import centroid_features
    >>> from maad.util import (power2dB, format_features, overlay_rois, plot2d,
                               overlay_centroid)
     
    Load audio and compute spectrogram 
     
    >>> s, fs = load('../data/spinetail.wav') 
    >>> Sxx,tn,fn,ext = spectrogram(s, fs, db_range=80) 
    >>> Sxx = power2dB(Sxx, db_range=80)
     
    Load annotations and plot
    
    >>> from maad.util import read_audacity_annot
    >>> rois = read_audacity_annot('../data/spinetail.txt') 
    >>> rois = format_features(rois, tn, fn) 
    >>> ax, fig = plot2d (Sxx, extent=ext)
    >>> ax, fig = overlay_rois(Sxx,rois, extent=ext, ax=ax, fig=fig)
    
    Compute the centroid of each rois, format to get results in the 
    temporal and spectral domain and overlay the centroids.
     
    >>> centroid = centroid_features(Sxx, rois) 
    >>> centroid = format_features(centroid, tn, fn)
    >>> ax, fig = overlay_centroid(Sxx,centroid, extent=ext, ax=ax, fig=fig)
    
    """     
     
    # Check input data 
    if type(Sxx) is not np.ndarray and len(Sxx.shape) != 2: 
        raise TypeError('Sxx must be an numpy 2D array')  
        
    # Convert the spectrogram in linear scale
    # This is necessary because we want to obtain the 90th percentile of the 
    # the energy inside each bbox.
    # if the spectrogram is a clean spectrogram, this is directly the SNR
    Sxx = maad.util.dB2power(Sxx)
     
    # check rois 
    if rois is not None: 
        if not(('min_t' and 'min_f' and 'max_t' and 'max_f') in rois): 
            raise TypeError('Array must be a Pandas DataFrame with column names: min_t, min_f, max_t, max_f. Check example in documentation.') 
     
    centroid=[] 
    area = []   
    snr = []
    if rois is None: 
        centroid = ndimage.center_of_mass(Sxx) 
        centroid = pd.DataFrame(np.asarray(centroid)).T 
        centroid.columns = ['centroid_y', 'centroid_x'] 
        centroid['area_xy'] = Sxx.shape[0] * Sxx.shape[1]
        centroid['duration_x'] = Sxx.shape[1]
        centroid['bandwidth_y'] = Sxx.shape[0]
        # TODO : add in MAAD
        centroid['snr'] = mean_dB(add_dB(Sxx,axis=0)) 
    else: 
        if im_rois is not None : 
            # real centroid and area
            rprops = measure.regionprops(im_rois, intensity_image=Sxx)
            centroid = [roi.weighted_centroid for roi in rprops]
            area = [roi.area for roi in rprops]
            # TODO : add in MAAD
            snr = [power2dB(np.mean(np.sum(roi.image_intensity,axis=0))) for roi in rprops]
        else:
            # rectangular area (overestimation) 
            area = (rois.max_y -rois.min_y) * (rois.max_x -rois.min_x)  
            # centroid of rectangular roi
            for _, row in rois.iterrows() : 
                row = pd.DataFrame(row).T 
                im_blobs = maad.rois.rois_to_imblobs(np.zeros(Sxx.shape), row)     
                rprops = measure.regionprops(im_blobs, intensity_image=Sxx)
                centroid.append(rprops.pop().weighted_centroid) 
                # TODO : add in MAAD
                snr.append(power2dB(np.mean(np.sum(rprops.pop().image_intensity,axis=0)))) 
                
        centroid = pd.DataFrame(centroid, columns=['centroid_y', 'centroid_x'], index=rois.index)
        
        ##### Energy of the signal (99th percentile of the bbox)
        centroid['snr'] = snr
        # ##### duration in number of pixels 
        centroid['duration_x'] = (rois.max_x -rois.min_x)  
        # ##### bandwidth in number of pixels 
        centroid['bandwidth_y'] = (rois.max_y -rois.min_y) 
        ##### area
        centroid['area_xy'] = area      
     
        # concat rois and centroid dataframes 
        centroid = rois.join(pd.DataFrame(centroid, index=rois.index))  

    return centroid 

#%%
###############################################################################
def extract_rois_in_soundscape(
    sig,
    params=cfg.PARAMS_EXTRACT,
    display=False,
    verbose=False,
    **kwargs):
    """ Extract all Rois in the audio file
    Parameters
    ----------
    sig : TYPE
        DESCRIPTION.
    params : dictionnary
        contains all the parameters to extract the rois 
    display : TYPE, optional
        DESCRIPTION. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.
    Returns
    -------
    df_rois : TYPE
        DESCRIPTION.
    """

    FUSION_ROIS = params['FUSION_ROIS']
    REMOVE_ROIS_FMIN_LIM = params['REMOVE_ROIS_FMIN_LIM']
    REMOVE_ROIS_FMAX_LIM = params['REMOVE_ROIS_FMAX_LIM']
    REMOVE_RAIN = params['REMOVE_RAIN']
    MIN_EVENT_DUR = params['MIN_EVENT_DUR'] # Minimum time duration of an event (in s)
    MAX_EVENT_DUR = params['MAX_EVENT_DUR'] # Minimum time duration of an event (in s)
    MIN_FREQ_BW = params['MIN_FREQ_BW'] # Minimum frequency bandwidth (in Hz)
    MAX_FREQ_BW = params['MAX_FREQ_BW']
    if (MIN_EVENT_DUR is not None) and (MIN_FREQ_BW is not None):  
        MIN_ROI_AREA = MIN_EVENT_DUR * MIN_FREQ_BW 
    else :
        MIN_ROI_AREA = None
    if (MAX_EVENT_DUR is not None) and (MAX_FREQ_BW is not None):  
        MAX_ROI_AREA = MAX_EVENT_DUR * MAX_FREQ_BW 
    else :
        MAX_ROI_AREA = None
    
    MAX_YX_RATIO = params['MAX_RATIO_YX']
    
    SEED_LEVEL = params["SEED_LEVEL"]
    LOW_LEVEL = params["LOW_LEVEL"]


    # 1. compute the spectrogram
    Sxx, tn, fn, ext = maad.sound.spectrogram(
                                sig,
                                params["SAMPLE_RATE"],
                                nperseg=params["NFFT"],
                                noverlap=params["NFFT"] // 2,
                                flims=[params["LOW_FREQ"], params["HIGH_FREQ"]])

    DELTA_T = tn[1] - tn[0]
    DELTA_F = fn[1] - fn[0]

    if verbose:
        print("\ntime resolution {}s".format(DELTA_T))
        print("frequency resolution {}s".format(DELTA_F))

    if display:
        # creating grid for subplots
        fig = plt.figure()
        fig.set_figheight(5)
        fig.set_figwidth(13)
        ax0 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=1)
        ax1 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=1)
        ax2 = plt.subplot2grid(shape=(2, 4), loc=(0, 1), colspan=1)
        ax3 = plt.subplot2grid(shape=(2, 4), loc=(1, 1), colspan=1)
        ax4 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), rowspan=2, colspan=2)

        maad.util.plot_wave(sig, fs=params["SAMPLE_RATE"], ax=ax0)

        maad.util.plot_spectrogram(
            Sxx,
            extent=ext,
            ax=ax1,
            title="1. original spectrogram",
            interpolation=None,
            now=False)
        
    ### 1. convert to dB
    Sxx_dB = maad.util.power2dB(Sxx, db_range=96) + 96
    
    ### 2. Clean spectrogram : remove background)

    if REMOVE_RAIN: 
        # remove single vertical lines
        Sxx_clean_dB, _ = maad.sound.remove_background_along_axis(Sxx_dB.T,
                                                            mode='mean',
                                                            N=1,
                                                            display=False)
        Sxx_dB = Sxx_clean_dB.T

    # remove single horizontal lines
    Sxx_clean_dB, _ = maad.sound.remove_background_along_axis(Sxx_dB,
                                                    mode='median',
                                                    N=1,
                                                    display=False)
    # Set the background to 0
    Sxx_clean_dB[Sxx_clean_dB<=0] = 0

    if display:
        maad.util.plot_spectrogram(
            Sxx_clean_dB,
            extent=ext,
            log_scale=False,
            ax=ax2,
            title="2. cleaned spectrogram",
            interpolation='none',
            now=False,
            vmin = 0,
            vmax = np.percentile(Sxx_clean_dB,99.9)
        )
    
    ### 4. Global snr estimation to threshold the spectrogram
    _,bgn,snr,_,_,_ = maad.sound.spectral_snr(maad.util.dB2power(Sxx_clean_dB))
    if verbose :
        print('BGN {}dB / SNR {}dB'.format(bgn,snr))
        
    ### 5. binarization of the spectrogram to select part of the spectrogram with
    # acoustic activity
    # Both parameters can be adapted to the situation in order to take more
    # or less ROIs that are more or less large        
    
    # TODO : add in roi and maybe in maad
    im_mask = maad.rois.create_mask(
        Sxx_clean_dB,
        mode_bin="absolute",
        # bin_h=SEED_LEVEL+snr,
        # bin_l=LOW_LEVEL+snr
        bin_h=SEED_LEVEL,
        bin_l=LOW_LEVEL
    )

    if display:
        maad.util.plot_spectrogram(
            im_mask,
            extent=ext,
            ax=ax3,
            title="3. mask",
            interpolation=None,
            now=False,
        )

    ### 6. Fusion ROIS    
    if type(FUSION_ROIS) is tuple :
        Ny_elements = round(FUSION_ROIS[0] / DELTA_T)
        Nx_elements = round(FUSION_ROIS[1] / DELTA_F)
        im_mask = closing(im_mask, footprint=np.ones([Nx_elements,Ny_elements]))

    # get the mask with rois (im_rois) and the bounding box for each rois (rois_bbox) 
    # and an unique index for each rois => in the pandas dataframe rois
    im_rois, df_rois  = maad.rois.select_rois(
        im_mask,
        min_roi=MIN_ROI_AREA, 
        max_roi=MAX_ROI_AREA)
    
    if verbose:
        print("BEFORE MERGING FOUND {} ROIS ".format(len(df_rois)))

    ### 7. Add centroids after merging
    if len(df_rois) >0 :
        # format ROis to initial tn and fn
        df_rois = maad.util.format_features(df_rois, tn, fn)
        
        # found the centroid and add the centroid parameters ('centroid_y',
        # 'centroid_x', 'duration_x', 'bandwidth_y', 'area_xy') into df_rois
        df_rois = _centroid_features(Sxx_clean_dB, df_rois, im_rois)
        
        # format ROis to initial tn and fn
        df_rois = maad.util.format_features(df_rois, tn, fn)
    
        ### 8. add a column ratio_xy 
        df_rois['ratio_xy'] = (df_rois.max_y -df_rois.min_y) / (df_rois.max_x -df_rois.min_x) 

        ### 9. remove some ROIs based on duration and bandwidth
        if MIN_EVENT_DUR is not None :
            df_rois = df_rois[df_rois['duration_t'] >= MIN_EVENT_DUR]
        if MAX_EVENT_DUR is not None :    
            df_rois = df_rois[df_rois['duration_t'] <= MAX_EVENT_DUR]

        # remove min and max frequency bandwidth 
        if MIN_FREQ_BW is not None :
            df_rois = df_rois[df_rois['bandwidth_f'] >= MIN_FREQ_BW]
        if MAX_FREQ_BW is not None :    
            df_rois = df_rois[df_rois['bandwidth_f'] <= MAX_FREQ_BW]
    
    ### 10. Remove some ROIS   
    if len(df_rois) >0 :
        if REMOVE_ROIS_FMIN_LIM is not None:
            low_frequency_threshold_in_pixels=None
            high_frequency_threshold_in_pixels=None

            if isinstance(REMOVE_ROIS_FMIN_LIM, (float, int)) :
                low_frequency_threshold_in_pixels = max(1, round(REMOVE_ROIS_FMIN_LIM / DELTA_F))
            elif isinstance(REMOVE_ROIS_FMIN_LIM, (tuple, list, np.ndarray)) and len(REMOVE_ROIS_FMIN_LIM) == 2 :
                low_frequency_threshold_in_pixels = max(1, round(REMOVE_ROIS_FMIN_LIM[0] / DELTA_F))
                high_frequency_threshold_in_pixels = min(im_rois.shape[1]-1, round(REMOVE_ROIS_FMIN_LIM[1] / DELTA_F))
            else:
                raise ValueError ('REMOVE_ROIS_FMAX_LIM should be {None, a single value or a tuple of 2 values')

            # retrieve the list of labels that match the condition
            list_labelID = df_rois[df_rois['min_y']<=low_frequency_threshold_in_pixels]['labelID']
            # set to 0 all the pixel that match the labelID that we want to remove
            for labelID in list_labelID.astype(int).tolist() :
                im_rois[im_rois==labelID] = 0
            # delete the rois corresponding to the labelID that we removed in im_mask
            df_rois = df_rois[~df_rois['labelID'].isin(list_labelID)]

            if high_frequency_threshold_in_pixels is not None :
                # retrieve the list of labels that match the condition  
                list_labelID = df_rois[df_rois['min_y']>=high_frequency_threshold_in_pixels]['labelID']
                # set to 0 all the pixel that match the labelID that we want to remove
                for labelID in list_labelID.astype(int).tolist() :
                    im_rois[im_rois==labelID] = 0
                # delete the rois corresponding to the labelID that we removed in im_mask
                df_rois = df_rois[~df_rois['labelID'].isin(list_labelID)]

        if REMOVE_ROIS_FMAX_LIM is not None:
            if isinstance(REMOVE_ROIS_FMAX_LIM, (float, int)) :
                high_frequency_threshold_in_pixels = min(im_rois.shape[1]-1, round(REMOVE_ROIS_FMAX_LIM / DELTA_F))
            else:
                raise ValueError ('REMOVE_ROIS_FMAX_LIM should be {None, or single value')

            # retrieve the list of labels that match the condition  
            list_labelID = df_rois[df_rois['max_y']>=high_frequency_threshold_in_pixels]['labelID']
            # set to 0 all the pixel that match the labelID that we want to remove
            for labelID in list_labelID.astype(int).tolist() :
                im_rois[im_rois==labelID] = 0
            # delete the rois corresponding to the labelID that we removed in im_mask
            df_rois = df_rois[~df_rois['labelID'].isin(list_labelID)]
        
        if MAX_YX_RATIO is not None:
            df_rois = df_rois[df_rois['ratio_xy'] < MAX_YX_RATIO]  

    # 11. Display the result
        if verbose:
            print("=> AFTER MERGING FOUND {} ROIS".format(len(df_rois)))
        
        if display:
            # Convert in dB
            X = maad.util.power2dB(Sxx, db_range=96) + 96
            kwargs.update({"vmax": np.max(X)})
            kwargs.update({"vmin": np.min(X)})
            kwargs.update({"ms": 4, "marker": "+"})
            maad.util.plot_spectrogram(X, ext, log_scale=False, title="5. Overlay ROIs", interpolation=None, now=False, ax=ax4, vmax=np.max(X), vmin=np.min(X))
            maad.util.overlay_rois(X, df_rois, edge_color='yellow', ax=ax4, now=True, **kwargs)
            print("=> number of ROIS after ploting {}".format(len(df_rois)))
            maad.util.overlay_centroid(X, df_rois, extent=ext, ax=ax4, fig=fig)
            print("=> number of ROIS after ploting {}".format(len(df_rois)))
        

    return df_rois