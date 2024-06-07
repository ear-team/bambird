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
# def _select_rois(im_bin, min_roi=None ,max_roi=None, 
#                 margins=(0,0), 
#                 verbose=False, display=False, savefig = None, **kwargs): 
#     """ 
#     Select regions of interest based on its dimensions.
    
#     The input is a binary mask, and the output is an image with labelled pixels. 
 
#     Parameters 
#     ---------- 
#     im : 2d ndarray of scalars 
#         Spectrogram (or image) 
         
#     min_roi, max_roi : scalars, optional, default : None 
#         Define the minimum and the maximum area possible for an ROI. If None,  
#         the minimum ROI area is 1 pixel and the maximum ROI area is the area of  
#         the image 
        
#     margins : tuple, default : (0, 0)
#         Before selected the ROIs, an optional closing (dilation followed by an
#         erosion) is performed on the binary mask. The element used for the closing
#         is defined my margins. The first number is the number of pixels along 
#         y axis (frequency) while the second number is the number of pixels along 
#         x axis (time). This operation will merge events that are closed to
#         each other in order to create a bigger ROIs encompassing all of them
        
#     verbose : boolean, optional, default is False
#         print messages
         
#     display : boolean, optional, default is False 
#         Display the signal if True 
         
#     savefig : string, optional, default is None 
#         Root filename (with full path) is required to save the figures. Postfix 
#         is added to the root filename. 
         
#     \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
            
#         - savefilename : str, optional, default :'_spectro_after_noise_subtraction.png' 
#             Postfix of the figure filename 
         
#         - figsize : tuple of integers, optional, default: (4,10) 
#             width, height in inches.   
         
#         - title : string, optional, default : 'Spectrogram' 
#             title of the figure 
         
#         - xlabel : string, optional, default : 'Time [s]' 
#             label of the horizontal axis 
         
#         - ylabel : string, optional, default : 'Amplitude [AU]' 
#             label of the vertical axis 
         
#         - cmap : string or Colormap object, optional, default is 'gray' 
#             See https://matplotlib.org/examples/color/colormaps_reference.html 
#             in order to get all the  existing colormaps 
#             examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic',  
#             'viridis'... 
         
#         - vmin, vmax : scalar, optional, default: None 
#             `vmin` and `vmax` are used in conjunction with norm to normalize 
#             luminance data.  Note if you pass a `norm` instance, your 
#             settings for `vmin` and `vmax` will be ignored. 
         
#         - extent : scalars (left, right, bottom, top), optional, default: None 
#             The location, in data-coordinates, of the lower-left and 
#             upper-right corners. If `None`, the image is positioned such that 
#             the pixel centers fall on zero-based (row, column) indices. 
         
#         - dpi : integer, optional, default is 96 
#             Dot per inch.  
#             For printed version, choose high dpi (i.e. dpi=300) => slow 
#             For screen version, choose low dpi (i.e. dpi=96) => fast 
         
#         - format : string, optional, default is 'png' 
#             Format to save the figure 
             
#         ... and more, see matplotlib    
 
#     Returns 
#     ------- 
#     im_rois: 2d ndarray 
#         image with labels as values 
             
#     rois: pandas DataFrame 
#         Regions of interest with future descriptors will be computed. 
#         Array have column names: ``labelID``, ``label``, ``min_y``, ``min_x``,
#         ``max_y``, ``max_x``,
#         Use the function ``maad.util.format_features`` before using 
#         centroid_features to format of the ``rois`` DataFrame 
#         correctly.
        
#     Examples 
#     -------- 
    
#     Load audio recording compute the spectrogram in dB.
    
#     >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
#     >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs, fcrop=(0,20000), display=True)           
#     >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
#     Smooth the spectrogram
    
#     >>> Sxx_dB_blurred = maad.sound.smooth(Sxx_dB)
    
#      Using image binarization, detect isolated region in the time-frequency domain with high density of energy, i.e. regions of interest (ROIs).
    
#     >>> im_bin = maad.rois.create_mask(Sxx_dB_blurred, bin_std=1.5, bin_per=0.5, mode='relative')
    
#     Select ROIs from the binary mask.
    
#     >>> im_rois, df_rois = maad.rois.select_rois(im_bin, display=True)
    
#     We detected the background noise as a ROI, and that multiple ROIs are mixed in a single region. To have better results, it is adviced to preprocess the spectrogram to remove the background noise before creating the mask.
    
#     >>> Sxx_noNoise = maad.sound.median_equalizer(Sxx)
#     >>> Sxx_noNoise_dB = maad.util.power2dB(Sxx_noNoise)     
#     >>> Sxx_noNoise_dB_blurred = maad.sound.smooth(Sxx_noNoise_dB)        
#     >>> im_bin2 = maad.rois.create_mask(Sxx_noNoise_dB_blurred, bin_std=6, bin_per=0.5, mode='relative') 
#     >>> im_rois2, df_rois2 = maad.rois.select_rois(im_bin2, display=True)
    
#     """ 
 
#     # test if max_roi and min_roi are defined 
#     if max_roi is None:  
#         # the maximum ROI is set to the aera of the image 
#         max_roi=im_bin.shape[0]*im_bin.shape[1] 
         
#     if min_roi is None: 
#         # the min ROI area is set to 1 pixel 
#         min_roi = 1 
    
#     if verbose :
#         print(72 * '_') 
#         print('Automatic ROIs selection in progress...') 
#         print ('**********************************************************') 
#         print ('  Min ROI area %d pix² | Max ROI area %d pix²' % (min_roi, max_roi)) 
#         print ('**********************************************************') 
        
#     # merge ROIS
#     if sum(margins) !=0 :
#         footprint = np.ones((margins[0]*2+1,margins[1]*2+1))
#         im_bin = closing(im_bin, footprint)            
 
#     labels = measure.label(im_bin)    #Find connected components in binary image 
#     rprops = measure.regionprops(labels) 
     
#     rois_bbox = [] 
#     rois_label = [] 
     
#     for roi in rprops: 
         
#         # select the rois  depending on their size 
#         if (roi.area >= min_roi) & (roi.area <= max_roi): 
#             # get the label 
#             rois_label.append(roi.label) 
#             # get rectangle coordonates            
#             rois_bbox.append (roi.bbox)     
                 
#     im_rois = np.isin(labels, rois_label)    # test if the indice is in the matrix of indices 
#     im_rois = im_rois* labels 
     
#     # create a list with labelID and labelName (None in this case) 
#     rois_label = list(zip(rois_label,['unknown']*len(rois_label))) 
    
#     # test if there is a roi
#     if len(rois_label)>0 :
#         # create a dataframe rois containing the coordonates and the label 
#         rois = np.concatenate((np.asarray(rois_label), np.asarray(rois_bbox)), axis=1) 
#         rois = pd.DataFrame(rois, columns = ['labelID', 'label', 'min_y','min_x','max_y', 'max_x']) 
#         # force type to integer 
#         rois = rois.astype({'label': str,'min_y':int,'min_x':int,'max_y':int, 'max_x':int}) 
#         # compensate half-open interval of bbox from skimage 
#         rois.max_y -= 1 
#         rois.max_x -= 1 
        
#     else :
#         rois = []    
#         rois = pd.DataFrame(rois, columns = ['labelID', 'label', 'min_y','min_x','max_y', 'max_x']) 
#         rois = rois.astype({'label': str,'min_y':int,'min_x':int,'max_y':int, 'max_x':int}) 
     
#     # Display 
#     if display :  
#         ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
#         xlabel =kwargs.pop('xlabel','Time [sec]')  
#         title  =kwargs.pop('title','Selected ROIs')  
#         extent=kwargs.pop('extent',None)
                
#         if extent is None : 
#             xlabel = 'pseudotime [points]'
#             ylabel = 'pseudofrequency [points]'
         
#         # randcmap = rand_cmap(len(rois_label)) 
#         # cmap   =kwargs.pop('cmap',randcmap)  
#         cmap   =kwargs.pop('cmap','tab20') 
         
#         _, fig = maad.util.plot2d (
#                                 im_rois, 
#                                 extent = extent,
#                                 title  = title,  
#                                 ylabel = ylabel, 
#                                 xlabel = xlabel,
#                                 cmap   = cmap, 
#                                 **kwargs) 
#         # SAVE FIGURE 
#         if savefig is not None :  
#             dpi   =kwargs.pop('dpi',96) 
#             format=kwargs.pop('format','png')  
#             filename=kwargs.pop('filename','_spectro_selectrois')                 
#             filename = savefig+filename+'.'+format 
#             fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format, 
#                         **kwargs)  
            
#     return im_rois, rois 

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

    # Sxx_clean_dB, _ = maad.sound.remove_background_along_axis(Sxx_dB,
    #                                                 mode=params["MODE_RMBCKG"],
    #                                                 N=params["N_RUNNING_MEAN"],
    #                                                 display=False)

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