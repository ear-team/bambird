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

# Scikit-Maad (ecoacoustics functions) package
import maad

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
                 'MASK_PARAM1': 26,
                 'MASK_PARAM2': 10,
                 'MAX_RATIO_YX': 7,
                 'MIN_DURATION': 0.1,
                 'MARGIN_T_LEFT': 0.2,
                 'MARGIN_T_RIGHT': 0.2,
                 'MARGIN_F_TOP': 250,
                 'MARGIN_F_BOTTOM': 250,
                 'MARGIN_T': 0.1,
                 'MARGIN_F': 250,
                 'FILTER_ORDER': 5}

#%%

###############################################################################
def _select_rois(im_bin, 
                 min_roi=None,
                 max_roi=None, 
                 margins=(0,0), 
                 verbose=False, 
                 display=False, 
                 **kwargs): 
    """ 
    Select regions of interest based on its dimensions.
    
    The input is a binary mask, and the output is an image with labelled pixels. 
 
    Parameters 
    ---------- 
    im : 2d ndarray of scalars 
        Spectrogram (or image) 
         
    min_roi, max_roi : scalars, optional, default : None 
        Define the minimum and the maximum area possible for an ROI. If None,  
        the minimum ROI area is 1 pixel and the maximum ROI area is the area of  
        the image 
        
    margins : tuple, default : (0, 0)
        Before selected the ROIs, an optional closing (dilation followed by an
        erosion) is performed on the binary mask. The element used for the closing
        is defined my margins. The first number is the number of pixels along 
        y axis (frequency) while the second number is the number of pixels along 
        x axis (time). This operation will merge events that are closed to
        each other in order to create a bigger ROIs encompassing all of them
        
    verbose : boolean, optional, default is False
        print messages
         
    display : boolean, optional, default is False 
        Display the signal if True 
                  
    \*\*kwargs, optional. This parameter is used by plt.plot  functions 
                     
        - figsize : tuple of integers, optional, default: (4,10) 
            width, height in inches.   
         
        - title : string, optional, default : 'Spectrogram' 
            title of the figure 
         
        - xlabel : string, optional, default : 'Time [s]' 
            label of the horizontal axis 
         
        - ylabel : string, optional, default : 'Amplitude [AU]' 
            label of the vertical axis 
         
        - cmap : string or Colormap object, optional, default is 'gray' 
            See https://matplotlib.org/examples/color/colormaps_reference.html 
            in order to get all the  existing colormaps 
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic',  
            'viridis'... 
         
        - vmin, vmax : scalar, optional, default: None 
            `vmin` and `vmax` are used in conjunction with norm to normalize 
            luminance data.  Note if you pass a `norm` instance, your 
            settings for `vmin` and `vmax` will be ignored. 
         
        - extent : scalars (left, right, bottom, top), optional, default: None 
            The location, in data-coordinates, of the lower-left and 
            upper-right corners. If `None`, the image is positioned such that 
            the pixel centers fall on zero-based (row, column) indices. 
         
        - dpi : integer, optional, default is 96 
            Dot per inch.  
            For printed version, choose high dpi (i.e. dpi=300) => slow 
            For screen version, choose low dpi (i.e. dpi=96) => fast 
             
        ... and more, see matplotlib    
 
    Returns 
    ------- 
    im_rois: 2d ndarray 
        image with labels as values 
             
    rois: pandas DataFrame 
        Regions of interest with future descriptors will be computed. 
        Array have column names: ``labelID``, ``label``, ``min_y``, ``min_x``,
        ``max_y``, ``max_x``,
        Use the function ``maad.util.format_features`` before using 
        centroid_features to format of the ``rois`` DataFrame 
        correctly.
        
    Examples 
    -------- 
    
    Load audio recording compute the spectrogram in dB.
    
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs, fcrop=(0,20000), display=True)           
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Smooth the spectrogram
    
    >>> Sxx_dB_blurred = maad.sound.smooth(Sxx_dB)
    
     Using image binarization, detect isolated region in the time-frequency domain with high density of energy, i.e. regions of interest (ROIs).
    
    >>> im_bin = maad.rois.create_mask(Sxx_dB_blurred, bin_std=1.5, bin_per=0.5, mode='relative')
    
    Select ROIs from the binary mask.
    
    >>> im_rois, df_rois = maad.rois.select_rois(im_bin, display=True)
    
    We detected the background noise as a ROI, and that multiple ROIs are mixed in a single region. To have better results, it is adviced to preprocess the spectrogram to remove the background noise before creating the mask.
    
    >>> Sxx_noNoise = maad.sound.median_equalizer(Sxx)
    >>> Sxx_noNoise_dB = maad.util.power2dB(Sxx_noNoise)     
    >>> Sxx_noNoise_dB_blurred = maad.sound.smooth(Sxx_noNoise_dB)        
    >>> im_bin2 = maad.rois.create_mask(Sxx_noNoise_dB_blurred, bin_std=6, bin_per=0.5, mode='relative') 
    >>> im_rois2, df_rois2 = maad.rois.select_rois(im_bin2, display=True)
    
    """ 
 
    # test if max_roi and min_roi are defined 
    if max_roi is None:  
        # the maximum ROI is set to the aera of the image 
        max_roi=im_bin.shape[0]*im_bin.shape[1] 
         
    if min_roi is None: 
        # the min ROI area is set to 1 pixel 
        min_roi = 1 
    
    if verbose :
        print(72 * '_') 
        print('Automatic ROIs selection in progress...') 
        print ('**********************************************************') 
        print ('  Min ROI area %d pix² | Max ROI area %d pix²' % (min_roi, max_roi)) 
        print ('**********************************************************') 
        
    # merge ROIS
    if sum(margins) !=0 :
        footprint = np.ones((margins[0]*2+1,margins[1]*2+1))
        im_bin = closing(im_bin, footprint)            
 
    labels = measure.label(im_bin)    #Find connected components in binary image 
    rprops = measure.regionprops(labels) 
     
    rois_bbox = [] 
    rois_label = [] 
     
    for roi in rprops: 
         
        # select the rois  depending on their size 
        if (roi.area >= min_roi) & (roi.area <= max_roi): 
            # get the label 
            rois_label.append(roi.label) 
            # get rectangle coordonates            
            rois_bbox.append (roi.bbox)     
                 
    im_rois = np.isin(labels, rois_label)    # test if the indice is in the matrix of indices 
    im_rois = im_rois* labels 
     
    # create a list with labelID and labelName (None in this case) 
    rois_label = list(zip(rois_label,['unknown']*len(rois_label))) 
    
    # test if there is a roi
    if len(rois_label)>0 :
        # create a dataframe rois containing the coordonates and the label 
        rois = np.concatenate((np.asarray(rois_label), np.asarray(rois_bbox)), axis=1) 
        rois = pd.DataFrame(rois, columns = ['labelID', 'label', 'min_y','min_x','max_y', 'max_x']) 
        # force type to integer 
        rois = rois.astype({'label': str,'min_y':int,'min_x':int,'max_y':int, 'max_x':int}) 
        # compensate half-open interval of bbox from skimage 
        rois.max_y -= 1 
        rois.max_x -= 1 
        
    else :
        rois = []    
        rois = pd.DataFrame(rois, columns = ['labelID', 'label', 'min_y','min_x','max_y', 'max_x']) 
        rois = rois.astype({'label': str,'min_y':int,'min_x':int,'max_y':int, 'max_x':int}) 
     
    # Display 
    if display :  
        ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
        xlabel =kwargs.pop('xlabel','Time [sec]')  
        title  =kwargs.pop('title','Selected ROIs')  
        extent=kwargs.pop('extent',None)
                
        if extent is None : 
            xlabel = 'pseudotime [points]'
            ylabel = 'pseudofrequency [points]'
         
        # randcmap = rand_cmap(len(rois_label)) 
        # cmap   =kwargs.pop('cmap',randcmap)  
        cmap   =kwargs.pop('cmap','tab20') 
         
        _, fig = maad.util.plot2d (
                                im_rois, 
                                extent = extent,
                                title  = title,  
                                ylabel = ylabel, 
                                xlabel = xlabel,
                                cmap   = cmap, 
                                **kwargs) 
            
    return im_rois, rois 

#%%
###############################################################################
def extract_rois_full_sig(
    sig,
    params=PARAMS_EXTRACT,
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

    # 1. compute the spectrogram
    Sxx, tn, fn, ext = maad.sound.spectrogram(
                                sig,
                                params["SAMPLE_RATE"],
                                nperseg=params["NFFT"],
                                noverlap=params["NFFT"] // 2,
                                flims=[params["LOW_FREQ"], params["HIGH_FREQ"]])

    t_resolution = tn[1] - tn[0]
    f_resolution = fn[1] - fn[0]

    if verbose:
        print("time resolution {}s".format(t_resolution))
        print("frequency resolution {}s".format(f_resolution))

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
        
    # 3. convert to dB
    Sxx_dB = maad.util.power2dB(Sxx, db_range=96) + 96
    
    # 2. Clean spectrogram : remove background)
    Sxx_clean_dB, _ = maad.sound.remove_background_along_axis(Sxx_dB,
                                                      mode=params["MODE_RMBCKG"],
                                                      N=params["N_RUNNING_MEAN"],
                                                      display=False)

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
    
    # 4. snr estimation to threshold the spectrogram
    _,bgn,snr,_,_,_ = maad.sound.spectral_snr(maad.util.dB2power(Sxx_clean_dB))
    if verbose :
        print('BGN {}dB / SNR {}dB'.format(bgn,snr))
        
    # 5. binarization of the spectrogram to select part of the spectrogram with
    # acoustic activity
    # Both parameters can be adapted to the situation in order to take more
    # or less ROIs that are more or less large        
    
    im_mask = maad.rois.create_mask(
        Sxx_clean_dB,
        mode_bin="absolute",
        # bin_h= snr + params["MASK_PARAM1"],
        # bin_l= snr + params["MASK_PARAM2"]
        bin_h=  params["MASK_PARAM1"],
        bin_l=  params["MASK_PARAM2"]
    )

    if display:
        maad.util.plot_spectrogram(
            im_mask,
            extent=ext,
            ax=ax3,
            title="3. mask",
            interpolation=None,
            now=True,
        )

    # 6. get the mask with rois (im_rois) and the bounding box for each rois (rois_bbox)
    # and an unique index for each rois => in the pandas dataframe rois
    margins = (round(params["MARGIN_F_BOTTOM"] / f_resolution),
               round(params["MARGIN_T_LEFT"] / t_resolution)) 
    im_rois, df_rois = _select_rois(im_mask, min_roi=None, margins = margins)
    
    # and format ROis to initial tn and fn
    df_rois = maad.util.format_features(df_rois, tn, fn)

    # 6Bis. found the centroid and add the centroid parameters ('centroid_y',
    # 'centroid_x', 'duration_x', 'bandwidth_y', 'area_xy') into df_rois
    df_rois =  maad.features.centroid_features(Sxx_clean_dB, df_rois, im_rois)
    
    # and format ROis to initial tn and fn
    df_rois = maad.util.format_features(df_rois, tn, fn)
    
    # Test if we found an ROI otherwise we pass to the next chunk
    if len(df_rois) > 0:    

        # 7. Remove ROIs with problems in the coordinates
        df_rois = df_rois[df_rois.min_x < df_rois.max_x]
        df_rois = df_rois[df_rois.min_y < df_rois.max_y]
        
        # 8. remove rois with ratio >max_ratio_xy (they are mostly artefact 
        # such as wind, rain or clipping)
        # add ratio x/y
        df_rois['ratio_yx'] = (df_rois.max_y -df_rois.min_y) / (df_rois.max_x -df_rois.min_x) 
        if params["MAX_RATIO_YX"] is not None :
            df_rois = df_rois[df_rois['ratio_yx'] < params["MAX_RATIO_YX"]] 
    
        # Drop two columns
        df_rois = df_rois.drop(columns=["labelID", "label"])

        # Keep only events with duration longer than MIN_DURATION
        df_rois = df_rois[((df_rois["max_t"]-df_rois["min_t"])>params["MIN_DURATION"])]
        
        # 8. remove rois with ratio >max_ratio_xy (they are mostly artefact 
        # such as wind, ain or clipping)
        # add ratio x/y
        df_rois['ratio_yx'] = (df_rois.max_y -df_rois.min_y) / (df_rois.max_x -df_rois.min_x) 
        
        if params["MAX_RATIO_YX"] is not None :
            df_rois = df_rois[df_rois['ratio_yx'] < params["MAX_RATIO_YX"]] 

        if verbose:
            print("=> AFTER MERGING FOUND {} ROIS".format(len(df_rois)))
        
        if display:
            # Convert in dB
            X = maad.util.power2dB(Sxx, db_range=96) + 96
            kwargs.update({"vmax": np.max(X)})
            kwargs.update({"vmin": np.min(X)})
            kwargs.update({"extent": ext})
            kwargs.update({"figsize": (1, 2.5)})
            maad.util.plot_spectrogram(
                X, ext, log_scale=False, ax=ax4, title="5. Overlay ROIs"
            )
            maad.util.overlay_rois(X, df_rois,
                              edge_color='yellow',
                              ax=ax4, fig=fig, **kwargs)
            kwargs.update(
                {"ms": 4, "marker": "+", "fig": fig, "ax": ax4})
            # ax, fig = maad.util.overlay_centroid(X, df_rois, **kwargs)
            fig.suptitle(kwargs.pop("suptitle", ""))
            fig.tight_layout()
    
    return df_rois