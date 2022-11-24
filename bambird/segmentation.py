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

# scikit-image (image processing) package
from skimage.transform import resize

# audio package
import librosa

# Parallel processing packages
from functools import partial
from tqdm import tqdm
from concurrent import futures

# scikit-image package
from skimage import measure
from skimage.morphology import closing

# Scikit-Maad (ecoacoustics functions) package
import maad

#
from bamxc import config as cfg
from bamxc import grab_audio_to_df

#%%

""" ===========================================================================

                    Functions to extract ROIS    

============================================================================"""


###############################################################################
def _select_rois(im_bin, min_roi=None ,max_roi=None, 
                margins=(0,0), 
                verbose=False, display=False, savefig = None, **kwargs): 
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
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
            
        - savefilename : str, optional, default :'_spectro_after_noise_subtraction.png' 
            Postfix of the figure filename 
         
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
         
        - format : string, optional, default is 'png' 
            Format to save the figure 
             
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
        # SAVE FIGURE 
        if savefig is not None :  
            dpi   =kwargs.pop('dpi',96) 
            format=kwargs.pop('format','png')  
            filename=kwargs.pop('filename','_spectro_selectrois')                 
            filename = savefig+filename+'.'+format 
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format, 
                        **kwargs)  
            
    return im_rois, rois 


###############################################################################
def _intersection_bbox(bb1, bb2):
    """
    test if 2 bounding-boxes intersect.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    boolean

    """

    # test if values are consistant (min < max)
    assert bb1["min_x"] <= bb1["max_x"]
    assert bb1["min_y"] <= bb1["max_y"]
    assert bb2["min_x"] <= bb2["max_x"]
    assert bb2["min_y"] <= bb2["max_y"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["min_x"], bb2["min_x"])
    y_top = max(bb1["min_y"], bb2["min_y"])
    x_right = min(bb1["max_x"], bb2["max_x"])
    y_bottom = min(bb1["max_y"], bb2["max_y"])

    if (x_right < x_left) or (y_bottom < y_top):
        is_intersected = False
    else:
        is_intersected = True

    return is_intersected

###############################################################################
def _fusion_bbox(df_rois, idx, idx2):
    """[fusion beginning and ends of 2 events that are too close to each other. It is applied if
    the 2 beginnings or the 2 ends are too close, or if the end of the first one is too close to the beginning
    of the second event. This function test all the events that follow each other in the dataframe.]

    Args:
        df_rois : 
            dataframe with all bbox to test for merging
        idx1 : 
            position of the FIRST bbox in the dataframe
        idx2 : 
            position of the SECOND bbox in the dataframe

    Return       df_rois: 
            returns the original dataframe with merged bbox
    """
    bb1 = df_rois.loc[idx]
    bb2 = df_rois.loc[idx2]

    min_t = min(bb1.min_t, bb2.min_t)
    max_t = max(bb1.max_t, bb2.max_t)
    min_f = min(bb1.min_f, bb2.min_f)
    max_f = max(bb1.max_f, bb2.max_f)

    df_rois.loc[idx, ["min_f", "min_t", "max_f", "max_t"]] = [
        min_f,
        min_t,
        max_f,
        max_t,
    ]

    min_x = min(bb1.min_x, bb2.min_x)
    max_x = max(bb1.max_x, bb2.max_x)
    min_y = min(bb1.min_y, bb2.min_y)
    max_y = max(bb1.max_y, bb2.max_y)

    df_rois.loc[idx, ["min_y", "min_x", "max_y", "max_x"]] = [
        min_y,
        min_x,
        max_y,
        max_x,
    ]

    df_rois = df_rois.drop(index=idx2)

    return df_rois

###############################################################################
def _merge_bbox(df_rois, margins):
    """[Merge two bbox that are within the margin. As the process is iterative
    if the resulting bbox is also close to another bbox, then it is merged. 
    And so on]

    Args:
        df_rois : 
            dataframe with all bbox to test for merging
        margins : 
            array with margins before/after the bbox and upper/lower the bbox

    Returns:
        df_rois: 
            returns the original dataframe with merged bbox
    """

    # loop until all bbox are merged
    # test if the number of rois is stable. if not, then run again the merging
    # process.
    num_rois = len(df_rois) + 1
    while len(df_rois) < num_rois:
        num_rois = len(df_rois)
        for idx, row in df_rois.iterrows():
            # test if the index idx is still in the dataframe
            if idx in df_rois.index:
                # remove the current ROIs (to avoid auto-merge)
                df_rois_without_current_row = df_rois.drop(idx)
                # add the margins to the bbox
                bb1 = row[["min_x", "min_y", "max_x", "max_y"]] + margins
                # loop to all the other ROIs
                for idx2, row2 in df_rois_without_current_row.iterrows():
                    # add the margins to the bbox
                    bb2 = row2[["min_x", "min_y", "max_x", "max_y"]] + margins
                    # If intersection => merge
                    if _intersection_bbox(bb1, bb2):
                        df_rois = _fusion_bbox(df_rois, idx, idx2)
                    # test if length of df_rois >1
                    if len(df_rois) <= 1:
                        break

    return df_rois




###############################################################################
def _save_rois(
    chunk,
    fs,
    df_rois,
    save_path,
    margins=(0.02, 50),
    filter_order=5,
    display=False):
    """ save the rois (= chunk) as new wave file. Margins around the bbox
    can  be applied around the bbox in order to enlarge the save bbox in time
    and frequency
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
def extract_rois_core(
    sig,
    params=cfg.DEFAULT_PARAMS_EXTRACT,
    display=False,
    verbose=False,
    **kwargs):
    
    """ Extract all Rois in the audio file
    Parameters
    ----------
    audio_path : TYPE
        DESCRIPTION. 
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
    
    # 1. compute the spectrogram
    Sxx, tn, fn, ext = maad.sound.spectrogram(
                                    sig,
                                    params["SAMPLE_RATE"],
                                    nperseg=params["NFFT"],
                                    noverlap=params["NFFT"] // 2)

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

        maad.util.plot_spectrogram(
                        Sxx,
                        extent=ext,
                        ax=ax0,
                        title="1. original spectrogram",
                        interpolation=None,
                        now=False)

    # 2. reduce the size of the spectrogram Sxx
    # Both parameters can be adapted to the audio signature that we want to select
    Sxx_reduced = resize(
                    Sxx, 
                    (Sxx.shape[0] // params["FACTOR_F"], 
                     Sxx.shape[1] // params["FACTOR_T"]),
                    anti_aliasing=True)
    
    if display:
        maad.util.plot_spectrogram(
                            Sxx_reduced,
                            log_scale=True,
                            extent=ext,
                            ax=ax1,
                            title="2. Resized spectrogram",
                            interpolation=None,
                            now=False)

    # 3. Clean spectrogram : remove background)
    Sxx_clean_reduced, _ = maad.sound.remove_background_along_axis(
                                                    Sxx_reduced,
                                                    mode=params["MODE_RMBCKG"],
                                                    N=params["N_RUNNING_MEAN"],
                                                    display=False)
    
    # 4. convert to dB
    Sxx_clean_dB_reduced = (maad.util.power2dB(Sxx_clean_reduced, db_range=96) +96)

    if display:
        maad.util.plot_spectrogram(
                        Sxx_clean_dB_reduced,
                        extent=ext,
                        log_scale=False,
                        ax=ax2,
                        title="3. cleaned spectrogram",
                        interpolation=None,
                        now=False)

    # reduce time and frequential vectors
    if params["FACTOR_T"] % 2 == 0:  # is even
        tn_reduced = (tn[np.arange(params["FACTOR_T"] // 2, 
                                   len(tn), 
                                   params["FACTOR_T"])]
                      + t_resolution/2)
    else:  # is odd
        tn_reduced = tn[np.arange(params["FACTOR_T"] // 2,
                                  len(tn), 
                                  params["FACTOR_T"])]

    if params["FACTOR_F"] % 2 == 0:  # is even
        fn_reduced = (fn[ np.arange(params["FACTOR_F"] // 2, 
                                    len(fn), 
                                    params["FACTOR_F"])]
                      + f_resolution/2)
    else:  # is odd
        fn_reduced = fn[np.arange(params["FACTOR_F"] // 2,
                                  len(fn), 
                                  params["FACTOR_F"])]

    # 7. binarization of the spectrogram to select part of the spectrogram with
    # acoustic activity
    # Both parameters can be adapted to the situation in order to take more
    # or less ROIs that are more or less large
    im_mask = maad.rois.create_mask(
                        Sxx_clean_dB_reduced,
                        mode_bin="absolute",
                        bin_h=params["MASK_PARAM1"],
                        bin_l=params["MASK_PARAM2"])

    if display:
        maad.util.plot_spectrogram(
                        im_mask,
                        extent=ext,
                        ax=ax3,
                        title="4. mask",
                        interpolation=None,
                        now=True)

    # 8. get the mask with rois (im_rois) and the bounding box for each rois (rois_bbox)
    # and an unique index for each rois => in the pandas dataframe rois
    _, df_rois = maad.rois.select_rois(im_mask, min_roi=None)
 
    # and format ROis to reduced tn and fn
    df_rois = maad.util.format_features(df_rois, tn_reduced, fn_reduced)
    # and format ROis to initial tn and fn
    df_rois = maad.util.format_features(df_rois, tn, fn)

    # Remove cut ROIs (begining and end of audio) and 1 line rois (vertical
    # and horizontal )
    # Drop ROIs with same min_x max_x and min_y max_y (=> 1 line)
    df_rois = df_rois[df_rois.min_x < df_rois.max_x]
    df_rois = df_rois[df_rois.min_y < df_rois.max_y]

    # Drop two columns
    df_rois = df_rois.drop(
        columns=["labelID", "label"])

    if verbose:
        print("\nBEFORE MERGING FOUND {} ROIS ".format(len(df_rois)))

    # Test if we found an ROI otherwise we pass to the next chunk
    if len(df_rois) > 0:

        if len(df_rois) > 1:
            # 9. Merge ROIs that are very close to each other.
            # Default, if marings = [0,0,0,0], bbox must overlap in order
            # to be merge. If margins is not null, all bbox are expanded
            # according to margins. If the expanded bbox overlapped, they
            # are merged.
            # The process is iterative which means than several ROIs can be
            # merged after several pass
            margins = [
                -round(params["MARGIN_T_LEFT"] / t_resolution),
                -round(params["MARGIN_F_BOTTOM"] / f_resolution),
                 round(params["MARGIN_T_RIGHT"] / t_resolution),
                 round(params["MARGIN_F_TOP"] / f_resolution)]
            df_rois = _merge_bbox(df_rois, margins)

        # Keep only events with duration longer than MIN_DURATION
        df_rois = df_rois[((df_rois["max_t"] - df_rois["min_t"]) > params["MIN_DURATION"])]

        if verbose:
            print("=> AFTER MERGING FOUND {} ROIS".format(len(df_rois)))

        if display:
            # Convert in dB
            X = maad.util.power2dB(Sxx, db_range=96) + 96
            kwargs = {"vmax": np.max(X)}
            kwargs.update({"vmin": np.min(X)})
            kwargs.update({"extent": ext})
            kwargs.update({"figsize": (1, 2.5)})
            maad.util.plot_spectrogram(X, ext, 
                                  log_scale=False, 
                                  ax=ax4, 
                                  title="5. Overlay ROIs")
            maad.util.overlay_rois(X, df_rois, 
                              ax=ax4, fig=fig,
                              edge_color='yellow',
                              **kwargs)
            kwargs.update({"ms": 4, "marker": "+", "fig": fig, "ax": ax4})
            # ax, fig = maad.util.overlay_centroid(X, df_rois, **kwargs)
            fig.suptitle(kwargs.pop("suptitle", ""))
            fig.tight_layout()

    return df_rois

###############################################################################
def extract_rois_full_sig(
    sig,
    params=cfg.DEFAULT_PARAMS_EXTRACT,
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
    _, df_rois = _select_rois(im_mask, min_roi=None, margins = margins)
    
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
        print()
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


###############################################################################
def single_file_extract_rois(
    audio_path,
    fun=extract_rois_full_sig,
    params=cfg.DEFAULT_PARAMS_EXTRACT,
    save_path=None,
    display=False,
    verbose=False):
    """ Extract all Rois in the audio file
    Parameters
    ----------
    audio_path : TYPE
        DESCRIPTION.
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
    # extract species and code from fullfilename 
    try :
        _, species = os.path.split(path)
        code = ((species.split(" ", 1)[0][0:3]).lower() + 
                (species.split(" ", 1)[1][0:3]).lower())
    except :
        _, species = os.path.split(path)
        code = species
    
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

        # Split signal into SIGNAL_LENGTH second chunks
        # Just like we did before (well, this could actually be a seperate function)
        # We also adjust the overlap (OVLP, ratio) between each chunk
        sig_splits = []
        n_points_per_chunk = int(params["SIGNAL_LENGTH"]* params["SAMPLE_RATE"])
        for i in range(0,len(sig), int(n_points_per_chunk * (1 - params["OVLP"]))):

            split = sig[i: i + n_points_per_chunk]

            # End of signal?
            if len(split) < n_points_per_chunk :
                break

            sig_splits.append(split)

        # print outputs
        if verbose:
            print("\n================================================================")
            print("SPECIES : {}".format(species))
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
                                    kwarg={'suptitle': filename + " " + species})
                
                # test if at least 1 ROI was found
                if len(current_df_rois) > 0 :
                    
                    # Add global information about the file and species for 
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
                    current_df_rois.insert(4, "code", code)
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
                seconds += params["SIGNAL_LENGTH"] * (1 - params["OVLP"])

    except Exception as e:
        if verbose:
            print("\nERROR : " + str(e))
        df_rois = pd.DataFrame()

    return df_rois

###############################################################################
def multicpu_extract_rois(
    dataset, # directory or csv or dataframe
    fun=extract_rois_full_sig,
    params=cfg.DEFAULT_PARAMS_EXTRACT,
    save_path=None,
    save_csv_filename='rois.csv',
    overwrite=False,
    nb_cpu=None,
    verbose=True):
    
    if verbose :
        print('======================= EXTRACT ROIS =========================\n')

    # if Dataset is a directory : 
    # > all audio files in the directory will be processed.
    #-------------------------------------------------------------------------
    
    # test if dataset_path is a valid path with audio files (wav or mp3)
    if isinstance(dataset, pd.DataFrame) == False :
        if os.path.isdir(dataset):
            
            # format dataset to Path
            dataset = Path(dataset)

            # create a dataframe with all recordings in the directory
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
    #-------------------------------------------------------------------------     
    elif isinstance(dataset, pd.DataFrame): 
        df_data = dataset.copy()
        
        # set default save_path and save_filename
        if save_path is None:
            save_path = str(Path(df_data.fullfilename.iloc[0]).parent.parent) + "_ROIS"
            
    else:
        raise Exception(
            "WARNING: dataset must be a valid path to a directory, a csv file"
            + "or a dataframe"
        )
        
    #----------------------------------------------------------------------------
    # Check if the output directory already exists
    dir_exist = os.path.exists(save_path) 
    if (dir_exist == False) or ((dir_exist == True) and (overwrite == True)) : 
        if (dir_exist == True) and (overwrite == True):
            if verbose:
                print(("The directory {} already exists" +
                      " and will be overwritten").format(save_path))   
        
        #----------------------------------------------------------------------
        try :
            # load the dataframe with all ROIs already extracted
            df_rois = pd.read_csv(save_path / save_csv_filename, 
                                  sep=';')
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
                print('   -number of species : %2.0f' % len(df_data[~mask].species.unique()))
                print('   -unique species code : {}'.format(df_data[~mask]['code'].unique()))
        
            # Number of CPU used for the calculation. By default, set to all available
            # CPUs
            if nb_cpu is None:
                nb_cpu = os.cpu_count()
    
            # define a new function with fixed parameters to give to the multicpu pool 
            multicpu_func = partial(
                single_file_extract_rois,
                fun=fun,
                params=params,
                save_path=save_path,
                display=False,
                verbose=False,
            )
        
            # Multicpu process
            with tqdm(total=len(df_data[~mask])) as pbar:
                with futures.ProcessPoolExecutor(max_workers=nb_cpu-1) as pool:
                    for df_rois_temp in pool.map(
                        multicpu_func, df_data[~mask]["fullfilename"].to_list()
                    ):
                        pbar.update(1)
                        df_rois = df_rois.append(df_rois_temp)
            
            # sort filename for each species
            #---------------------------------
            df_rois_sorted = pd.DataFrame()
            for code in df_rois["code"].unique():
                df_rois_sorted = df_rois_sorted.append(
                    df_rois[df_rois["code"] == code].sort_index()
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
                print(('***WARNING** The csv file {0} does not exist in the directory where are the ROIS. \n' +
                      '====> Please, delete the directory {1} and restart the extraction process').format
                      (save_csv_filename, save_path))
                
        if verbose:
            print("The directory {} already exists".format(save_path))
            print(">>> EXTRACTION PROCESS ABORTED <<<")

    return df_rois_sorted, csv_fullfilename

